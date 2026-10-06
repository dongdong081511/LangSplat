#!/bin/bash
# EXP-048 restore: rebuild teatime/ramen/waldo a07 DINO fields (lost in cleanup accident)
# Smooth features and base ckpts intact in dataset/; figurines a07 untouched.
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

run_scene () {
  SC=$1; LF=$2; PORT=$3
  CKPT=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
  echo "=== restore $SC a07 $(date +%H:%M:%S)"
  for L in 1 2 3; do
    if [ ! -f output/${SC}_dino_32ds_a07_$L/chkpnt30000.pth ]; then
      $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32ds_a07 \
        --language_features_name $LF --feature_level $L --include_feature \
        --start_checkpoint $CKPT --port $PORT 2>&1 | tail -1
      ls output/${SC}_dino_32ds_a07_$L/chkpnt30000.pth || { echo TRAIN_FAIL $SC L$L; exit 1; }
    fi
    RD=output/${SC}_dino_32ds_a07_$L/train/ours_None/renders_npy
    if [ ! -f $RD/00000.npy ]; then
      $PY -u render.py -m output/${SC}_dino_32ds_a07_$L --include_feature 2>&1 | tail -1
      ls $RD/00000.npy || { echo RENDER_FAIL $SC L$L; exit 1; }
    fi
  done
  echo "=== $SC restored $(date +%H:%M:%S)"
}

run_scene teatime language_features_dim32_dino_smooth_a07 6111
run_scene ramen language_features_dim32_dino_smooth_a07 6112
run_scene waldo_kitchen language_features_dim32_dino_smooth_a07 6113
echo "EXP-048 restore ALL DONE $(date +%H:%M:%S)"
