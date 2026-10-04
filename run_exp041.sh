#!/bin/bash
# EXP-041: McNemar significance (re-render 24 fields, 8 evals with per-pair JSON, paired test)
# EXP-042: teatime alpha sweep (raw / a03 / a05, a07 exists) with per-pair JSON
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
mkdir -p eval_result/mcnemar
echo "==== EXP-041 start $(date +%H:%M:%S)"

patch_dim() {
  $PY - "$1" <<'PYEOF'
import re, sys
DIM = sys.argv[1]
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', f'#define NUM_CHANNELS_language_feature {DIM}'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  f'torch.zeros((self._xyz.shape[0], {DIM}), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  f'torch.zeros((means3D.shape[0], {DIM}), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print(f'patched to {DIM}')
PYEOF
  (cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_${1}_exp041.log 2>&1) || { echo COMPILE_FAIL_DIM_$1; tail -20 /tmp/compile_${1}_exp041.log; exit 1; }
  echo "rasterizer compiled to ${1}d $(date +%H:%M:%S)"
}

render_if_missing() {
  local D=$1
  if [ -d "$D/train/ours_None/renders_npy" ] && [ "$(ls $D/train/ours_None/renders_npy | wc -l)" -gt 100 ]; then
    echo "skip render $D (exists)"; return
  fi
  echo "--- render $D $(date +%H:%M:%S)"
  $PY -u render.py -m $D --include_feature 2>&1 | tail -1
  local N=$(ls $D/train/ours_None/renders_npy 2>/dev/null | wc -l)
  echo "renders: $N"
  [ "$N" -gt 100 ] || { echo RENDER_FAIL $D; exit 1; }
}

# ---------- EXP-041 renders: 32d a07 fields (4 scenes x 3) ----------
patch_dim 32
for S in waldo_kitchen teatime figurines ramen; do
  for L in 1 2 3; do render_if_missing output/${S}_dino_32ds_a07_$L; done
done
# ---------- 24d: figurines clip ----------
patch_dim 24
for L in 1 2 3; do render_if_missing output/figurines_24d_$L; done
# ---------- 8d: clip fields (waldo, teatime, ramen) ----------
patch_dim 8
for L in 1 2 3; do
  render_if_missing output/waldo_kitchen_clip8d_$L
  render_if_missing output/teatime_8d_$L
  render_if_missing output/ramen_clip8d_$L
done

# ---------- EXP-041 evals with per-pair JSON ----------
cd eval
EVAL() { echo "--- EVAL $1 $(date +%H:%M:%S)"; $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$2 \
  --scene_root ../dataset/lerf_ovs/$2 --db_render_dirs ../output/$3 ../output/$4 ../output/$5 \
  --query_subdir $6 --tag $1 --out_json ../eval_result/mcnemar/$1.json 2>&1 | grep -E "^\[" ; }

EVAL WALDO_CLIP8d       waldo_kitchen waldo_kitchen_clip8d_1        waldo_kitchen_clip8d_2        waldo_kitchen_clip8d_3        language_features_dim8
EVAL WALDO_DINO_a07     waldo_kitchen waldo_kitchen_dino_32ds_a07_1 waldo_kitchen_dino_32ds_a07_2 waldo_kitchen_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07
EVAL TEATIME_CLIP8d     teatime teatime_8d_1 teatime_8d_2 teatime_8d_3 language_features_dim8
EVAL TEATIME_DINO_a07   teatime teatime_dino_32ds_a07_1 teatime_dino_32ds_a07_2 teatime_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07
EVAL FIGURINES_CLIP24d  figurines figurines_24d_1 figurines_24d_2 figurines_24d_3 language_features_dim24
EVAL FIGURINES_DINO_a07 figurines figurines_dino_32ds_a07_1 figurines_dino_32ds_a07_2 figurines_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07
EVAL RAMEN_CLIP8d       ramen ramen_clip8d_1 ramen_clip8d_2 ramen_clip8d_3 language_features_dim8
EVAL RAMEN_DINO_a07     ramen ramen_dino_32ds_a07_1 ramen_dino_32ds_a07_2 ramen_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07

# ---------- McNemar paired tests (both protocols) ----------
for P in any first; do
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/WALDO_CLIP8d.json      --dino_json ../eval_result/mcnemar/WALDO_DINO_a07.json      --protocol $P --tag WALDO_$P
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/TEATIME_CLIP8d.json    --dino_json ../eval_result/mcnemar/TEATIME_DINO_a07.json    --protocol $P --tag TEATIME_$P
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/FIGURINES_CLIP24d.json --dino_json ../eval_result/mcnemar/FIGURINES_DINO_a07.json  --protocol $P --tag FIGURINES_$P
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/RAMEN_CLIP8d.json      --dino_json ../eval_result/mcnemar/RAMEN_DINO_a07.json      --protocol $P --tag RAMEN_$P
done
cd ..
echo "==== EXP-041 done $(date +%H:%M:%S)"

# ---------- EXP-042: teatime alpha sweep (needs 32d again) ----------
echo "==== EXP-042 start $(date +%H:%M:%S)"
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime
patch_dim 32
for L in 1 2 3; do render_if_missing output/teatime_dino_32d_$L; done

cd eval
echo "--- EVAL TEATIME_DINO_raw $(date +%H:%M:%S)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/teatime \
  --scene_root ../dataset/lerf_ovs/teatime \
  --db_render_dirs ../output/teatime_dino_32d_1 ../output/teatime_dino_32d_2 ../output/teatime_dino_32d_3 \
  --query_subdir language_features_dim32_dino --tag TEATIME_DINO_raw \
  --out_json ../eval_result/mcnemar/TEATIME_DINO_raw.json 2>&1 | grep -E "^\["

for A in 0.3 0.5; do
  TAG=a0${A#0.}   # 0.3 -> a03
  echo "--- smooth alpha=$A tag=$TAG $(date +%H:%M:%S)"
  if [ ! -d $SRC/language_features_dim32_dino_smooth_$TAG ]; then
    $PY -u smooth_tiles.py --scene teatime --alpha $A --iter_tag $TAG 2>&1 | tail -2
    $PY - <<PYEOF
import numpy as np, os
SRC = "$SRC"
o = np.load(f'{SRC}/language_features_dim32_dino/{sorted(os.listdir(SRC+"/language_features_dim32_dino"))[0]}')
s = np.load(f'{SRC}/language_features_dim32_dino_smooth_{TAG}/{sorted(os.listdir(SRC+"/language_features_dim32_dino_smooth_"+TAG))[0]}')
d = float(np.abs(s - o).max()); print('smooth maxdiff', d); assert d > 1e-3, 'SMOOTH_FAILED_SILENT'
PYEOF
  fi
  cd ..
  for L in 1 2 3; do
    [ -f output/teatime_dino_32ds_${TAG}_$L/chkpnt30000.pth ] && continue
    echo "--- train a-sweep $TAG level $L $(date +%H:%M:%S)"
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

# alpha curve summary + McNemar vs CLIP field for each alpha
$PY - <<'PYEOF'
import json
for tag in ['raw', 'a03', 'a05', 'a07']:
    p = f'../eval_result/mcnemar/TEATIME_DINO_{tag}.json'
    try:
        d = json.load(open(p))
    except Exception:
        print(f'alpha {tag}: missing'); continue
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
echo "==== EXP-041/042 ALL DONE $(date +%H:%M:%S)"
