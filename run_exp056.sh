#!/bin/bash
# EXP-056: LangSplat official 3d baseline on figurines/ramen/waldo_kitchen (teatime=EXP-001 0.6431)
# Table A baseline row: 3d AE, no-ensemble + ensemble dual report
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
echo "==== EXP-056 start $(date +%H:%M:%S)"

$PY - 3 <<'PYEOF'
import re, sys
DIM = sys.argv[1]
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', f'#define NUM_CHANNELS_language_feature {DIM}'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  f'torch.zeros((self._xyz.shape[0], {DIM}), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  f'torch.zeros((means3D.shape[0], {DIM}), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print(f'patched to {DIM}')
PYEOF
(cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_3d_exp056.log 2>&1) || { echo COMPILE_FAIL; tail -20 /tmp/compile_3d_exp056.log; exit 1; }
echo "rasterizer compiled to 3d $(date +%H:%M:%S)"

for SC in figurines ramen waldo_kitchen; do
  case $SC in
    figurines) THRESH=0.45 ;;
    ramen) THRESH=0.55 ;;
    waldo_kitchen) THRESH=0.40 ;;
  esac
  BASE=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
  echo "=== $SC 3d chain start $(date +%H:%M:%S)"
  for L in 1 2 3; do
    $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_3d \
      --language_features_name language_features_dim3 --feature_level $L \
      --include_feature --start_checkpoint $BASE --port 6174 2>&1 | tail -1
    ls output/${SC}_3d_$L/chkpnt30000.pth || { echo TRAIN_FAIL_${SC}_L$L; exit 1; }
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
      --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k $MODE 2>&1 | grep -E "mIoU|mAcc"
  done
  cd ..
  echo "=== $SC 3d done $(date +%H:%M:%S)"
done
echo "==== EXP056_ALL_DONE $(date +%H:%M:%S)"
