#!/bin/bash
# EXP-052: alpha=0.85 smoothing (3D term from a07 renders), teatime + figurines
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
run_scene () {
  SC=$1; FEAT=$2; ABBR=$3; PORT=$4
  $PY eval/smooth_tiles.py --scene $SC --alpha 0.85 --src_subdir $FEAT \
    --render_base ${SC}_dino_32ds_a07 --iter_tag a085 2>&1 | tail -1
  CKPT=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
  for L in 1 2 3; do
    $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32ds_a085 \
      --language_features_name ${FEAT}_smooth_a085 --feature_level $L \
      --include_feature --start_checkpoint $CKPT --port $PORT 2>&1 | tail -1
    $PY -u render.py -m output/${SC}_dino_32ds_a085_$L --include_feature 2>&1 | tail -1
  done
  cd eval
  $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC --scene_root ../dataset/lerf_ovs/$SC \
    --db_render_dirs ../output/${SC}_dino_32ds_a085_1 ../output/${SC}_dino_32ds_a085_2 ../output/${SC}_dino_32ds_a085_3 \
    --query_subdir ${FEAT}_smooth_a085 --tag ${SC}_A085 \
    --out_json ../eval_result/adaptive/exp052_${ABBR}_a085.json 2>&1 | grep -E "chosen|multi-frame"
  cd ..
  rm -rf output/${SC}_dino_32ds_a085_1 output/${SC}_dino_32ds_a085_2 output/${SC}_dino_32ds_a085_3
}
run_scene figurines language_features_dim32_dino FIG 6160
run_scene teatime language_features_dim32_dino TEA 6161
echo EXP052_ALL_DONE
