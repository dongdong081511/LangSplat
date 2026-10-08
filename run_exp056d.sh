#!/bin/bash
# EXP-056c: waldo_kitchen 3d fix chain (poll 056b then retrain+eval)
set -e
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
while ! grep -q "EXP056C_ALL_DONE" eval_result/exp056c.log 2>/dev/null; do sleep 60; done
echo "==== 056d start $(date +%H:%M:%S)"
rm -rf output/waldo_kitchen_3d_1 output/waldo_kitchen_3d_2 output/waldo_kitchen_3d_3
BASE=dataset/lerf_ovs/waldo_kitchen/output/waldo_kitchen_-1/chkpnt30000.pth
for L in 1 2 3; do
  $PY -u train.py -s dataset/lerf_ovs/waldo_kitchen -m output/waldo_kitchen_3d \
    --language_features_name language_features_dim3 --feature_level $L \
    --include_feature --start_checkpoint $BASE --port 6177 2>&1 | tail -1
  $PY -u render.py -m output/waldo_kitchen_3d_$L --include_feature 2>&1 | tail -1
  ls output/waldo_kitchen_3d_$L/train/ours_None/renders_npy/00000.npy || exit 1
done
cd eval
for MODE in "" "--prompt_ensemble"; do
  echo "--- waldo_kitchen 3d FIX eval ens=${MODE:-none} $(date +%H:%M:%S)"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name waldo_kitchen_3d \
    --clip_ae_ckpt ../autoencoder/ckpt/waldo_kitchen/best_ckpt.pth \
    --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
    --mask_thresh 0.40 --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k $MODE 2>&1 | tail -12 || echo EVAL_NONZERO
done
cd ..
echo "==== EXP056C_ALL_DONE $(date +%H:%M:%S)"
