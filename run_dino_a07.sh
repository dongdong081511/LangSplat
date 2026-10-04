#!/bin/bash
set -e; set -o pipefail
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/figurines
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
echo "==== EXP-037b smooth-a05 retrain start $(date +%H:%M:%S)"
for L in 1 2 3; do
  [ -f output/figurines_dino_32ds_a07_$L/chkpnt30000.pth ] && { echo "L$L ckpt exists, skip train"; continue; }
  echo "--- 32d GS level $L start $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/figurines_dino_32ds_a07 \
      --language_features_name language_features_dim32_dino_smooth_a07 \
      --feature_level $L --include_feature \
      --start_checkpoint $SRC/output/figurines_-1/chkpnt30000.pth \
      --port 5563$L 2>&1 | tail -2
  ls output/figurines_dino_32ds_a07_$L/chkpnt30000.pth
done

for L in 1 2 3; do
  $PY -u render.py -m output/figurines_dino_32ds_a07_$L --include_feature 2>&1 | tail -2
done
for L in 1 2 3; do
  echo "renders_npy 32d level $L: $(ls output/figurines_dino_32ds_a07_$L/train/ours_None/renders_npy | wc -l)"
done

cd eval
echo "--- image-query eval figurines DINO field $(date +%H:%M:%S)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/figurines \
    --scene_root ../dataset/lerf_ovs/figurines \
    --db_render_dirs ../output/figurines_dino_32ds_a07_1 ../output/figurines_dino_32ds_a07_2 ../output/figurines_dino_32ds_a07_3 \
    --query_subdir language_features_dim32_dino_smooth_a07 --tag DINO32d 2>&1 | grep -v Warning

echo "--- image-query eval CLIP 8d field (baseline)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/figurines \
    --scene_root ../dataset/lerf_ovs/figurines \
    --db_render_dirs ../output/figurines_24d_1 ../output/figurines_24d_2 ../output/figurines_24d_3 \
    --query_subdir language_features_dim24 --tag CLIP24d 2>&1 | grep -v Warning
cd ..
echo "==== figurines DINO field done $(date +%H:%M:%S) ===="
echo "ALL_EXP_DONE"
