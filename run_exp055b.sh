#!/bin/bash
# EXP-055b: render-mask query eval on 3 scenes with assets (teatime/ramen DINO a07 + waldo raw; CLIP sides)
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python

run_eval () {  # scene gtname dno_dir3 dno_sub clip_dir3 clip_sub tag
  local SC=$1 DNO=$2 DSUB=$3 CLP=$4 CSUB=$5 TAG=$6
  cd eval
  $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC --scene_root ../dataset/lerf_ovs/$SC \
    --db_render_dirs ../output/${DNO}_1 ../output/${DNO}_2 ../output/${DNO}_3 \
    --query_subdir $DSUB --tag ${TAG}_DINO_RM \
    --query_from_render --query_tile_mask \
    --out_json ../eval_result/adaptive/exp055_${TAG}_DINO_RM.json 2>&1 | grep -E "chosen|multi-frame"
  $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC --scene_root ../dataset/lerf_ovs/$SC \
    --db_render_dirs ../output/${CLP}_1 ../output/${CLP}_2 ../output/${CLP}_3 \
    --query_subdir $CSUB --tag ${TAG}_CLIP_RM \
    --query_from_render --query_tile_mask \
    --out_json ../eval_result/adaptive/exp055_${TAG}_CLIP_RM.json 2>&1 | grep -E "chosen|multi-frame"
  cd ..
}
run_eval teatime teatime_dino_32ds_a07 language_features_dim32_dino_smooth_a07 teatime_8d language_features_dim8 TEA
run_eval ramen ramen_dino_32ds_a07 language_features_dim32_dino_smooth_a07 ramen_clip8d language_features_dim8 RAM
run_eval waldo_kitchen waldo_kitchen_dino_32d language_features_dim32_dino waldo_kitchen_clip8d language_features_dim8 WAL
echo EXP055B_ALL_DONE
