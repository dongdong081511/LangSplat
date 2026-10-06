#!/bin/bash
# EXP-043 phase B: per-tile gated smoothing (cons signal, hard tau=0.85, beta=0.5)
# -> retrain 3 levels -> render -> image eval + McNemar (vs CLIP field and vs a07)
# Scene order puts the discriminating scene (waldo) first.
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
TAG=g07
echo "==== EXP-043b start $(date +%H:%M:%S)"

grep -q "NUM_CHANNELS_language_feature 32" submodules/langsplat-rasterization/cuda_rasterizer/config.h || { echo RASTERIZER_NOT_32D; exit 1; }

render_if_missing() {
  local D=$1
  if [ -d "$D/train/ours_None/renders_npy" ] && [ "$(ls $D/train/ours_None/renders_npy | wc -l)" -gt 100 ]; then
    echo "skip render $D (exists)"; return
  fi
  echo "--- render $D $(date +%H:%M:%S)"
  $PY -u render.py -m $D --include_feature 2>&1 | tail -1
  local N=$(ls $D/train/ours_None/renders_npy 2>/dev/null | wc -l)
  echo "renders: $N"; [ "$N" -gt 100 ] || { echo RENDER_FAIL $D; exit 1; }
}

for SC in waldo_kitchen teatime figurines ramen; do
  case $SC in teatime) ABBR=TEATIME;; figurines) ABBR=FIGURINES; CLIPJ=FIGURINES_CLIP24d.json;; waldo_kitchen) ABBR=WALDO; CLIPJ=WALDO_CLIP8d.json;; ramen) ABBR=RAMEN; CLIPJ=RAMEN_CLIP8d.json;; esac
  [ "$SC" = "teatime" ] && CLIPJ=TEATIME_CLIP8d.json

  echo "=== scene $SC ($ABBR) $(date +%H:%M:%S)"
  if [ ! -d dataset/lerf_ovs/$SC/language_features_dim32_dino_smooth_$TAG ]; then
    $PY -u eval/smooth_tiles_adaptive.py --scene $SC --alpha 0.7 --beta 0.5 \
        --signal cons --gate hard --tau 0.85 --iter_tag $TAG \
        --stats_out eval_result/adaptive/gated_$SC.json
  fi
  # guard: gate actually acted AND features changed vs source
  $PY - <<PYEOF
import json, numpy as np, glob
SC = "$SC"
st = json.load(open('eval_result/adaptive/gated_%s.json' % SC))
gl = [v['frac_gated_out'] for v in st['per_level'].values()]
ma = [v['mean_alpha'] for v in st['per_level'].values()]
print('frac_gated_out per level', gl, 'mean_alpha', ma)
assert max(gl) > 0.05, 'GATE_DID_NOTHING'
src = sorted(glob.glob('dataset/lerf_ovs/%s/language_features_dim32_dino/*_f.npy' % SC))[0]
new = sorted(glob.glob('dataset/lerf_ovs/%s/language_features_dim32_dino_smooth_g07/*_f.npy' % SC))[0]
d = float(np.abs(np.load(new) - np.load(src)).max())
print('smooth maxdiff', d)
assert d > 1e-3, 'SMOOTH_FAILED_SILENT'
PYEOF

  for L in 1 2 3; do
    [ -f output/${SC}_dino_32ds_${TAG}_$L/chkpnt30000.pth ] && continue
    echo "--- train $TAG level $L $(date +%H:%M:%S)"
    $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32ds_${TAG} \
        --language_features_name language_features_dim32_dino_smooth_${TAG} --feature_level $L --include_feature \
        --start_checkpoint dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth \
        --port 607$L 2>&1 | tail -2
    ls output/${SC}_dino_32ds_${TAG}_$L/chkpnt30000.pth || { echo TRAIN_FAIL $SC L$L; exit 1; }
  done
  for L in 1 2 3; do render_if_missing output/${SC}_dino_32ds_${TAG}_$L; done

  cd eval
  echo "--- EVAL ${ABBR}_DINO_g07 $(date +%H:%M:%S)"
  $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC \
    --scene_root ../dataset/lerf_ovs/$SC \
    --db_render_dirs ../output/${SC}_dino_32ds_${TAG}_1 ../output/${SC}_dino_32ds_${TAG}_2 ../output/${SC}_dino_32ds_${TAG}_3 \
    --query_subdir language_features_dim32_dino_smooth_${TAG} --tag ${ABBR}_DINO_g07 \
    --out_json ../eval_result/mcnemar/${ABBR}_DINO_g07.json 2>&1 | grep -E "^\["
  for P in first any; do
    $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/$CLIPJ \
      --dino_json ../eval_result/mcnemar/${ABBR}_DINO_g07.json --protocol $P --tag ${ABBR}_g07_vs_CLIP_$P
    $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/${ABBR}_DINO_a07.json \
      --dino_json ../eval_result/mcnemar/${ABBR}_DINO_g07.json --protocol $P --tag ${ABBR}_g07_vs_a07_$P
  done
  cd ..
done

$PY - <<'PYEOF'
import json
for ab in ['WALDO', 'TEATIME', 'FIGURINES', 'RAMEN']:
    try:
        d = json.load(open('eval_result/mcnemar/%s_DINO_g07.json' % ab))
    except Exception:
        print(ab, 'g07 missing'); continue
    hf = sum(x['hit_first'] for x in d['pairs']); ha = sum(x['hit_any'] for x in d['pairs'])
    n = d['n_pairs']
    print('%s g07: first %d/%d=%.2f%%  any %d/%d=%.2f%%' % (ab, hf, n, 100*hf/n, ha, n, 100*ha/n))
PYEOF
echo "==== EXP-043b ALL DONE $(date +%H:%M:%S)"
