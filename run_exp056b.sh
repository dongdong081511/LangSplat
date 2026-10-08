#!/bin/bash
# EXP-056b: fix json_folder path; skip finished figurines chains
set -e
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

for SC in figurines ramen waldo_kitchen; do
  case $SC in
    figurines) THRESH=0.45 ;;
    ramen) THRESH=0.55 ;;
    waldo_kitchen) THRESH=0.40 ;;
  esac
  BASE=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
  echo "=== $SC 3d chain start $(date +%H:%M:%S)"
  for L in 1 2 3; do
    if [ -f output/${SC}_3d_$L/chkpnt30000.pth ] && [ -f output/${SC}_3d_$L/train/ours_None/renders_npy/00000.npy ]; then
      echo "skip ${SC}_3d_$L (done)"; continue
    fi
    $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_3d \
      --language_features_name language_features_dim3 --feature_level $L \
      --include_feature --start_checkpoint $BASE --port 6174 2>&1 | tail -1
    $PY -u render.py -m output/${SC}_3d_$L --include_feature 2>&1 | tail -1
    ls output/${SC}_3d_$L/train/ours_None/renders_npy/00000.npy || { echo RENDER_FAIL_${SC}_L$L; exit 1; }
  done
  cd eval
  for MODE in "" "--prompt_ensemble"; do
    echo "--- $SC 3d eval ens=${MODE:-none} $(date +%H:%M:%S)"
    $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name ${SC}_3d \
      --clip_ae_ckpt ../autoencoder/ckpt/$SC/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh $THRESH --encoder_dims 256 128 64 32 3 \
      --decoder_dims 16 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k $MODE 2>&1 | tail -12 || echo EVAL_NONZERO_${SC}
  done
  cd ..
  echo "=== $SC 3d done $(date +%H:%M:%S)"
done
echo "==== EXP056B_ALL_DONE $(date +%H:%M:%S)"
