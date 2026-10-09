#!/bin/bash
# EXP-059: AE 64d (waldo native) → encode → 3D 训练×3 → 渲染 → eval + McNemar
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
SRC=dataset/lerf_ovs/waldo_kitchen
BASE=$SRC/output/waldo_kitchen_-1/chkpnt30000.pth
echo "==== EXP-059 chain start $(date +%H:%M:%S)"

# A. AE 64d (结构对称: 256 128 64 64 64 -> 64 128 256 256 768)
if [ ! -f ckpt/waldo_kitchen_dino64n/best_ckpt.pth ]; then
  echo "--- A AE 64d $(date +%H:%M:%S)"
  $PY -u autoencoder/train.py --dataset_path $SRC --data_subdir language_features_dino_native \
      --dataset_name waldo_kitchen_dino64n --input_dim 768 \
      --encoder_dims 256 128 64 64 64 --decoder_dims 64 128 256 256 768 2>&1 | tail -4
fi
ls ckpt/waldo_kitchen_dino64n/best_ckpt.pth

# B. encode dim64
if [ ! -d $SRC/language_features_dim64_dino_native ]; then
  echo "--- B encode $(date +%H:%M:%S)"
  $PY -u -m mm_langsplat.encode_dim3 --dataset_path $SRC --data_subdir language_features_dino_native \
      --out_subdir language_features_dim64_dino_native --ae_ckpt ckpt/waldo_kitchen_dino64n/best_ckpt.pth \
      --input_dim 768 --encoder_dims 256 128 64 64 64 --decoder_dims 64 128 256 256 768 2>&1 | tail -3
fi
ls $SRC/language_features_dim64_dino_native | wc -l

# C. 2D 快测 dim64 (AE 断点复诊: 目标 > 48.08%)
cd eval
$PY -u -c "
import sys; sys.path.insert(0, '.')
import tile_retrieval_test as trt
gt = trt.load_gt()
trt.run('../dataset/lerf_ovs/waldo_kitchen/language_features_dim64_dino_native', gt, 'dim64_native')
" 2>&1 | grep -E "===|TOTAL"
cd ..

# D. 3D 训练 ×3 (64d, 冒烟即首段训练)
for L in 1 2 3; do
  [ -f output/waldo_kitchen_dino_64dn_$L/chkpnt30000.pth ] && { echo "skip D L$L"; continue; }
  echo "--- D 3D train 64d level $L $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/waldo_kitchen_dino_64dn \
      --language_features_name language_features_dim64_dino_native --feature_level $L --include_feature \
      --start_checkpoint $BASE --port 619$L 2>&1 | tail -2
  ls output/waldo_kitchen_dino_64dn_$L/chkpnt30000.pth
done

# E. 渲染 ×3
for L in 1 2 3; do
  D=output/waldo_kitchen_dino_64dn_$L
  [ -d $D/train/ours_None/renders_npy ] && [ "$(ls $D/train/ours_None/renders_npy | wc -l)" -gt 100 ] && { echo "skip E L$L"; continue; }
  echo "--- E render level $L $(date +%H:%M:%S)"
  $PY -u render.py -m $D --include_feature 2>&1 | tail -1
done

# F. eval + McNemar
cd eval
echo "--- F eval $(date +%H:%M:%S)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/waldo_kitchen \
  --scene_root ../dataset/lerf_ovs/waldo_kitchen \
  --db_render_dirs ../output/waldo_kitchen_dino_64dn_1 ../output/waldo_kitchen_dino_64dn_2 ../output/waldo_kitchen_dino_64dn_3 \
  --query_subdir language_features_dim64_dino_native --tag WALDO_DINO_64native \
  --out_json ../eval_result/mcnemar/WALDO_DINO_64native.json 2>&1 | grep -E "^\[" || echo EVAL_WARN
for P in any first majority; do
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/WALDO_CLIP8d.json \
    --dino_json ../eval_result/mcnemar/WALDO_DINO_64native.json --protocol $P --tag WALDO_64native_$P 2>&1 | tail -1
done
cd ..
echo "==== EXP-059 ALL DONE $(date +%H:%M:%S)"
