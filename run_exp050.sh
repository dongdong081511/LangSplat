#!/bin/bash
# EXP-050: 3D field with struct-AE codes (teatime, raw DINO features, no smoothing)
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
SC=teatime; TAG=teatime_dino_32d_struct05; PORT=6150
CKPT=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
for L in 1 2 3; do
  $PY -u train.py -s dataset/lerf_ovs/$SC -m output/$TAG \
    --language_features_name language_features_dim32_dino_struct05 --feature_level $L \
    --include_feature --start_checkpoint $CKPT --port $PORT 2>&1 | tail -1
  ls output/${TAG}_$L/chkpnt30000.pth || { echo TRAIN_FAIL L$L; exit 1; }
  $PY -u render.py -m output/${TAG}_$L --include_feature 2>&1 | tail -1
  ls output/${TAG}_$L/train/ours_None/renders_npy/00000.npy || { echo RENDER_FAIL L$L; exit 1; }
done
cd eval
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC --scene_root ../dataset/lerf_ovs/$SC \
  --db_render_dirs ../output/${TAG}_1 ../output/${TAG}_2 ../output/${TAG}_3 \
  --query_subdir language_features_dim32_dino_struct05 --tag ${SC}_STRUCT05 \
  --out_json ../eval_result/adaptive/exp050_${SC}_struct05.json 2>&1 | grep -E "chosen|multi-frame"
echo EXP050_3D_DONE
