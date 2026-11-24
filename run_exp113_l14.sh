#!/bin/bash
# EXP-113 (P2-1): DINOv2 backbone-scale transfer ViT-B/14 -> ViT-L/14 (1024d).
# Stage A: re-extract tile features with dinov2_vitl14 (same SAM tiles; verify _s.npy
#          bit-identical to the deployed language_features_dino seg maps).
# Stage B: teatime L14 32d raw field full chain (AE 1024->32, encode, 3-level train, render).
# Stage C: exhaustive frame-pair probe (eval/exp113_dino_backbone.py) 2D + 3D endpoints.
set -e -o pipefail
cd /home/xiedexia/project/LangSplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export TORCH_CUDA_ARCH_LIST="8.9"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SCENES="teatime waldo_kitchen"

echo "==== EXP-113 start $(date +%H:%M:%S) ===="

# Stage A: L14 tile-feature extraction + assembly
for SCENE in $SCENES; do
  SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/$SCENE
  if ls $SRC/language_features_dino_l14/*_f.npy >/dev/null 2>&1; then
    echo "--- [$SCENE] L14 features already assembled, skip"
    continue
  fi
  echo "--- [$SCENE] L14 extraction start $(date +%H:%M:%S)"
  $PY -u preprocess.py --dataset_path $SRC --use_dino \
      --save_subdir language_features_l14work \
      --dino_model_name dinov2_vitl14 2>&1 | tail -3
  mkdir -p $SRC/language_features_dino_l14
  N_BAD=0
  for f in $SRC/language_features_l14work/*_f_dino.npy; do
    stem=$(basename $f _f_dino.npy)
    cp $f $SRC/language_features_dino_l14/${stem}_f.npy
    cp $SRC/language_features_l14work/${stem}_s.npy $SRC/language_features_dino_l14/${stem}_s.npy
    # verify tiles identical to deployed extraction: seg maps must match bit-wise
    if ! cmp -s $SRC/language_features_l14work/${stem}_s.npy $SRC/language_features_dino/${stem}_s.npy; then
      N_BAD=$((N_BAD+1))
    fi
  done
  echo "--- [$SCENE] assembled $(ls $SRC/language_features_dino_l14/*_f.npy | wc -l) frames, segmap mismatch=$N_BAD"
  rm -rf $SRC/language_features_l14work
done

# Stage B: teatime L14 32d raw field
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime
if [ ! -f ckpt/teatime_dino_l14/best_ckpt.pth ]; then
  echo "--- teatime L14 AE train $(date +%H:%M:%S)"
  $PY -u autoencoder/train.py --dataset_path $SRC \
      --data_subdir language_features_dino_l14 \
      --dataset_name teatime_dino_l14 --input_dim 1024 \
      --encoder_dims 256 128 32 32 32 \
      --decoder_dims 32 128 256 256 1024 2>&1 | tail -3
fi
if [ ! -d $SRC/language_features_dim32_dino_l14 ]; then
  echo "--- teatime L14 encode $(date +%H:%M:%S)"
  $PY -u -m mm_langsplat.encode_dim3 \
      --dataset_path $SRC \
      --data_subdir language_features_dino_l14 \
      --out_subdir language_features_dim32_dino_l14 \
      --ae_ckpt ckpt/teatime_dino_l14/best_ckpt.pth \
      --input_dim 1024 \
      --encoder_dims 256 128 32 32 32 \
      --decoder_dims 32 128 256 256 1024 2>&1 | tail -5
fi
for L in 1 2 3; do
  if [ ! -f output/teatime_dino_l14_$L/chkpnt30000.pth ]; then
    echo "--- teatime L14 field level $L start $(date +%H:%M:%S)"
    $PY -u train.py -s $SRC -m output/teatime_dino_l14_$L \
        --language_features_name language_features_dim32_dino_l14 \
        --feature_level $L --include_feature \
        --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
        --port 5580$L 2>&1 | tail -2
  fi
  $PY -u render.py -m output/teatime_dino_l14_$L --include_feature 2>&1 | tail -1
  echo "renders L$L: $(ls output/teatime_dino_l14_$L/train/ours_None/renders_npy | wc -l)"
done

# Stage C: probe (2D endpoints on both scenes + 3D endpoint where renders exist)
cd eval
/home/xiedexia/miniconda3/envs/laga/bin/python exp113_dino_backbone.py teatime,waldo_kitchen 2>&1 | tail -40
cd ..
echo "ALL_EXP113_DONE $(date +%H:%M:%S)"
