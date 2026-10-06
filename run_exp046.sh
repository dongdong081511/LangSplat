#!/bin/bash
# EXP-046: teatime tile similarity structure distillation (lf_rel_weight grid {1,5,20})
# Loss = L1 regression anchor + w_rel * MSE(S3, S2) where S = cosine similarity matrix
# over same-frame tiles (2D GT vs render tile means). Discriminative end-to-end field.
# Tag rule: model dir teatime_dino_32dr_r<W>_<L>, json TEATIME_DINO_cr<W>.json
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
SC=teatime
LF=language_features_dim32_dino
CKPT=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth

for W in 1 5 20; do
  TAG=r$W
  echo "=== $SC lf_rel w=$W tag=$TAG $(date +%H:%M:%S)"
  for L in 1 2 3; do
    if [ ! -f output/${SC}_dino_32d${TAG}_$L/chkpnt30000.pth ]; then
      $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32d${TAG} \
        --language_features_name $LF --feature_level $L --include_feature \
        --start_checkpoint $CKPT --lf_rel_weight $W --port 607$L 2>&1 | tail -2
      ls output/${SC}_dino_32d${TAG}_$L/chkpnt30000.pth || { echo TRAIN_FAIL w$W L$L; exit 1; }
    fi
  done
  for L in 1 2 3; do
    RD=output/${SC}_dino_32d${TAG}_$L/train/ours_None/renders_npy
    if [ ! -f $RD/00000.npy ]; then
      $PY -u render.py -m output/${SC}_dino_32d${TAG}_$L --include_feature 2>&1 | tail -1
      ls $RD/00000.npy || { echo RENDER_FAIL w$W L$L; exit 1; }
    fi
  done
  (cd eval && $PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC \
    --scene_root ../dataset/lerf_ovs/$SC \
    --db_render_dirs ../output/${SC}_dino_32d${TAG}_1 ../output/${SC}_dino_32d${TAG}_2 ../output/${SC}_dino_32d${TAG}_3 \
    --query_subdir $LF --tag TEATIME_DINO_c${TAG} \
    --out_json ../eval_result/mcnemar/TEATIME_DINO_c${TAG}.json 2>&1 | grep -E "^\[")
  for P in first any; do
    for REFJ in TEATIME_CLIP8d.json TEATIME_DINO_raw.json TEATIME_DINO_a07.json; do
      REF=$(basename $REFJ .json)
      $PY -u eval/mcnemar.py --clip_json eval_result/mcnemar/$REFJ \
        --dino_json eval_result/mcnemar/TEATIME_DINO_c${TAG}.json --protocol $P \
        --tag TEATIME_c${TAG}_vs_${REF}_$P
    done
  done
  echo "=== w=$W done $(date +%H:%M:%S)"
done
echo "EXP-046 teatime grid ALL DONE $(date +%H:%M:%S)"
