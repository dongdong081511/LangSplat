#!/bin/bash
# EXP-055a: figurines a09 retrain (cleaned post EXP-052b; features intact)
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
SC=figurines
CKPT=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
echo "=== fig a09 retrain start $(date +%H:%M:%S)"
for L in 1 2 3; do
  $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32ds_a09 \
    --language_features_name language_features_dim32_dino_smooth_a09 --feature_level $L \
    --include_feature --start_checkpoint $CKPT --port 6173 2>&1 | tail -1
  ls output/${SC}_dino_32ds_a09_$L/chkpnt30000.pth || { echo TRAIN_FAIL L$L; exit 1; }
  $PY -u render.py -m output/${SC}_dino_32ds_a09_$L --include_feature 2>&1 | tail -1
  ls output/${SC}_dino_32ds_a09_$L/train/ours_None/renders_npy/00000.npy || { echo RENDER_FAIL L$L; exit 1; }
done
cd eval
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC --scene_root ../dataset/lerf_ovs/$SC \
  --db_render_dirs ../output/${SC}_dino_32ds_a09_1 ../output/${SC}_dino_32ds_a09_2 ../output/${SC}_dino_32ds_a09_3 \
  --query_subdir language_features_dim32_dino_smooth_a09 --tag FIG_a09_RM \
  --query_from_render --query_tile_mask \
  --out_json ../eval_result/adaptive/exp055_FIG_DINO_RM.json 2>&1 | grep -E "chosen|multi-frame"
cd ..
echo "=== fig a09 done $(date +%H:%M:%S)"
echo EXP055A_ALL_DONE
