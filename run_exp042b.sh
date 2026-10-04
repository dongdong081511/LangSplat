#!/bin/bash
# EXP-042b: resume teatime alpha sweep from a03 training (smooth a03 features already on disk)
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
echo "==== EXP-042b start $(date +%H:%M:%S)"

# verify rasterizer is 32d (EXP-042 crashed after compiling 32d, before any other patch)
grep -q "NUM_CHANNELS_language_feature 32" submodules/langsplat-rasterization/cuda_rasterizer/config.h || { echo RASTERIZER_NOT_32D; exit 1; }

guard() {  # verify smoothed features actually differ from source (anti silent-failure)
  local TAG=$1
  local F1=$(ls $SRC/language_features_dim32_dino/*_f.npy | head -1 | xargs basename)
  local F2=$(ls $SRC/language_features_dim32_dino_smooth_${TAG}/*_f.npy | head -1 | xargs basename)
  $PY -c "
import numpy as np
o = np.load('$SRC/language_features_dim32_dino/$F1')
s = np.load('$SRC/language_features_dim32_dino_smooth_${TAG}/$F2')
d = float(np.abs(s - o).max()); print('smooth maxdiff', d)
assert d > 1e-3, 'SMOOTH_FAILED_SILENT'
"
}

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

cd eval
for A in 0.3 0.5; do
  TAG="a0${A#0.}"
  echo "--- alpha $A tag=$TAG $(date +%H:%M:%S)"
  [ -d $SRC/language_features_dim32_dino_smooth_$TAG ] || $PY -u smooth_tiles.py --scene teatime --alpha $A --iter_tag $TAG 2>&1 | tail -2
  guard $TAG
  cd ..
  for L in 1 2 3; do
    [ -f output/teatime_dino_32ds_${TAG}_$L/chkpnt30000.pth ] && continue
    echo "--- train $TAG level $L $(date +%H:%M:%S)"
    $PY -u train.py -s $SRC -m output/teatime_dino_32ds_${TAG} \
        --language_features_name language_features_dim32_dino_smooth_${TAG} --feature_level $L --include_feature \
        --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
        --port 607$L 2>&1 | tail -2
    ls output/teatime_dino_32ds_${TAG}_$L/chkpnt30000.pth
  done
  for L in 1 2 3; do render_if_missing output/teatime_dino_32ds_${TAG}_$L; done
  cd eval
  echo "--- EVAL TEATIME_DINO_$TAG $(date +%H:%M:%S)"
  $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/teatime \
    --scene_root ../dataset/lerf_ovs/teatime \
    --db_render_dirs ../output/teatime_dino_32ds_${TAG}_1 ../output/teatime_dino_32ds_${TAG}_2 ../output/teatime_dino_32ds_${TAG}_3 \
    --query_subdir language_features_dim32_dino_smooth_${TAG} --tag TEATIME_DINO_$TAG \
    --out_json ../eval_result/mcnemar/TEATIME_DINO_$TAG.json 2>&1 | grep -E "^\["
done

# alpha curve summary (first/any) + McNemar vs CLIP field per alpha
$PY - <<'PYEOF'
import json
for tag in ['raw', 'a03', 'a05', 'a07']:
    p = f'../eval_result/mcnemar/TEATIME_DINO_{tag}.json'
    try:
        d = json.load(open(p))
    except Exception:
        print(f'teatime alpha={tag}: missing'); continue
    hf = sum(x['hit_first'] for x in d['pairs']); ha = sum(x['hit_any'] for x in d['pairs'])
    n = d['n_pairs']
    print(f'teatime alpha={tag}: first {hf}/{n}={hf/n:.2%}  any {ha}/{n}={ha/n:.2%}')
PYEOF
for P in any first; do
  for T in raw a03 a05 a07; do
    J=../eval_result/mcnemar/TEATIME_DINO_${T}.json
    [ -f $J ] && $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/TEATIME_CLIP8d.json --dino_json $J --protocol $P --tag TEATIME_${T}_vs_CLIP_$P
  done
done
cd ..
echo "==== EXP-042b ALL DONE $(date +%H:%M:%S)"
