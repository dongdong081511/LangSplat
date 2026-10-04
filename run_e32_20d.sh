#!/bin/bash
# EXP-033: teatime e32 AE 20d full chain (v4 conventions: -m base name, default lr, 100 epochs)
set -e -o pipefail
cd /home/xiedexia/project/LangSplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export TORCH_CUDA_ARCH_LIST="8.9"
export PATH=/home/xiedexia/.conda/envs/langsplat/bin:$PATH
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime

echo "==== DIM=20 start $(date +%H:%M:%S) ===="
$PY - <<'EOF'
import re
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', '#define NUM_CHANNELS_language_feature 20'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  'torch.zeros((self._xyz.shape[0], 20), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  'torch.zeros((means3D.shape[0], 20), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read()
    s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'
    open(f, 'w').write(s2)
print('patched to 20')
EOF

cd submodules/langsplat-rasterization
rm -rf build
$PY setup.py build_ext --inplace > /tmp/compile_20d.log 2>&1 || { echo "COMPILE FAIL"; tail -30 /tmp/compile_20d.log; exit 1; }
cd ../..
echo "rasterizer compiled to 20 d"

$PY -u autoencoder/train.py --dataset_path $SRC \
    --data_subdir language_features_laion400m_e32 \
    --dataset_name teatime_e32_20d \
    --encoder_dims 256 128 64 32 20 \
    --decoder_dims 16 32 64 128 256 256 512 2>&1 | tail -3
ls ckpt/teatime_e32_20d/best_ckpt.pth

$PY -u -m mm_langsplat.encode_dim3 \
    --dataset_path $SRC \
    --data_subdir language_features_laion400m_e32 \
    --out_subdir language_features_dim20_e32 \
    --ae_ckpt ckpt/teatime_e32_20d/best_ckpt.pth \
    --encoder_dims 256 128 64 32 20 \
    --decoder_dims 16 32 64 128 256 256 512 2>&1 | tail -3
ls $SRC/language_features_dim20_e32/ | wc -l

for L in 1 2 3; do
  echo "--- 20d GS level $L start $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/teatime_e32_20d \
      --language_features_name language_features_dim20_e32 \
      --feature_level $L --include_feature \
      --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
      --port 5559$L 2>&1 | tail -2
  ls output/teatime_e32_20d_$L/chkpnt30000.pth
done

for L in 1 2 3; do
  $PY -u render.py -m output/teatime_e32_20d_$L --include_feature 2>&1 | tail -2
done
for L in 1 2 3; do
  echo "renders_npy 20d level $L: $(ls output/teatime_e32_20d_$L/train/ours_None/renders_npy | wc -l)"
done

ln -sfn teatime dataset/lerf_ovs/label/teatime_e32_20d
cd eval
for MODE in "" "--prompt_ensemble"; do
  echo "--- 20d eval mode=$MODE"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name teatime_e32_20d \
      --clip_ae_ckpt ../ckpt/teatime_e32_20d/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh 0.4 --encoder_dims 256 128 64 32 20 \
      --decoder_dims 16 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion400m_e32 $MODE 2>&1 | grep -E "mIoU|mAcc"
done
cd ..
echo "==== DIM=20 done $(date +%H:%M:%S) ===="
echo "ALL_EXP_DONE"
