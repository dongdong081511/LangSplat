#!/bin/bash
# EXP-043 phase A: re-render raw DINO fields (renders were cleaned) + per-tile
# signal distribution study (--dry, no features written). Gate threshold is
# decided offline from eval_result/adaptive/signals_*.json before phase B.
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
echo "==== EXP-043a start $(date +%H:%M:%S)"

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

# figurines raw level1 dir was cleaned in earlier cleanup; retrain it (level1
# gating signal + level1 tile smoothing both need its renders)
if [ ! -f output/figurines_dino_32d_1/chkpnt30000.pth ]; then
  echo "--- retrain figurines_dino_32d level1 $(date +%H:%M:%S)"
  $PY -u train.py -s dataset/lerf_ovs/figurines -m output/figurines_dino_32d \
      --language_features_name language_features_dim32_dino --feature_level 1 --include_feature \
      --start_checkpoint dataset/lerf_ovs/figurines/output/figurines_-1/chkpnt30000.pth \
      --port 6071 2>&1 | tail -2
  ls output/figurines_dino_32d_1/chkpnt30000.pth || { echo TRAIN_FAIL figurines_dino_32d_1; exit 1; }
fi

for SC in teatime figurines waldo_kitchen ramen; do
  for L in 1 2 3; do
    render_if_missing output/${SC}_dino_32d_$L
  done
done

mkdir -p eval_result/adaptive
for SC in teatime figurines waldo_kitchen ramen; do
  echo "--- signal study $SC $(date +%H:%M:%S)"
  $PY -u eval/smooth_tiles_adaptive.py --scene $SC --alpha 0.7 --iter_tag study \
      --signal cons --gate none --dry \
      --stats_out eval_result/adaptive/signals_$SC.json
done
echo "==== EXP-043a ALL DONE $(date +%H:%M:%S)"
