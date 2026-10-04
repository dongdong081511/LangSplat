#!/bin/bash
# EXP-040: any-bbox re-eval (teatime/figurines need re-render) + waldo alpha ablation (raw vs a07)
# Render order by rasterizer dim: 32d (waldo raw dino + teatime/figurines a07) -> 24d (figurines clip) -> 8d (teatime clip)
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
echo "==== EXP-040 start $(date +%H:%M:%S)"

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
  (cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_${1}_exp040.log 2>&1) || { echo COMPILE_FAIL_DIM_$1; tail -20 /tmp/compile_${1}_exp040.log; exit 1; }
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

for S in teatime figurines waldo_kitchen; do
  [ -d dataset/lerf_ovs/label/$S ] || { echo MISSING_GT $S; exit 1; }
done

# --- 1. 32d fields ---
patch_dim 32
for L in 1 2 3; do
  render_if_missing output/waldo_kitchen_dino_32d_$L
  render_if_missing output/teatime_dino_32ds_a07_$L
  render_if_missing output/figurines_dino_32ds_a07_$L
done

# --- 2. 24d field: figurines clip24d ---
patch_dim 24
for L in 1 2 3; do render_if_missing output/figurines_24d_$L; done

# --- 3. 8d field: teatime clip8d ---
patch_dim 8
for L in 1 2 3; do render_if_missing output/teatime_8d_$L; done

# --- 4. image evals (eval_image_query.py prints both first-bbox and any-bbox) ---
cd eval
EVAL() { echo "--- EVAL $1 $(date +%H:%M:%S)"; $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$2 \
  --scene_root ../dataset/lerf_ovs/$2 --db_render_dirs ../output/$3 ../output/$4 ../output/$5 \
  --query_subdir $6 --tag $1 2>&1 | grep -E "^\[" ; }

EVAL WALDO_CLIP8d       waldo_kitchen waldo_kitchen_clip8d_1       waldo_kitchen_clip8d_2       waldo_kitchen_clip8d_3       language_features_dim8
EVAL WALDO_DINO32d_raw  waldo_kitchen waldo_kitchen_dino_32d_1     waldo_kitchen_dino_32d_2     waldo_kitchen_dino_32d_3     language_features_dim32_dino
EVAL WALDO_DINO32d_a07  waldo_kitchen waldo_kitchen_dino_32ds_a07_1 waldo_kitchen_dino_32ds_a07_2 waldo_kitchen_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07
EVAL TEATIME_CLIP8d     teatime teatime_8d_1 teatime_8d_2 teatime_8d_3 language_features_dim8
EVAL TEATIME_DINO32d_a07 teatime teatime_dino_32ds_a07_1 teatime_dino_32ds_a07_2 teatime_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07
EVAL FIGURINES_CLIP24d  figurines figurines_24d_1 figurines_24d_2 figurines_24d_3 language_features_dim24
EVAL FIGURINES_DINO32d_a07 figurines figurines_dino_32ds_a07_1 figurines_dino_32ds_a07_2 figurines_dino_32ds_a07_3 language_features_dim32_dino_smooth_a07

cd ..
echo "==== EXP-040 ALL DONE $(date +%H:%M:%S)"
