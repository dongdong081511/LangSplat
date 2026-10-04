#!/bin/bash
# EXP-031/032 v4 recovery: -m uses base name (train.py appends _level)
set -e -o pipefail
cd /home/xiedexia/project/LangSplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export TORCH_CUDA_ARCH_LIST="8.9"
export PATH=/home/xiedexia/.conda/envs/langsplat/bin:$PATH
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime

# ---- DIM=12: reuse trained level1, train level2/3, render+eval ----
[ -d output/teatime_e32_12d_1 ] || mv output/teatime_e32_12d_1_1 output/teatime_e32_12d_1
ls output/teatime_e32_12d_1/chkpnt30000.pth

for L in 2 3; do
  echo "--- 12d GS level $L start $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/teatime_e32_12d \
      --language_features_name language_features_dim12_e32 \
      --feature_level $L --include_feature \
      --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
      --port 5559$L 2>&1 | tail -2
  ls output/teatime_e32_12d_$L/chkpnt30000.pth
done

for L in 1 2 3; do
  $PY -u render.py -m output/teatime_e32_12d_$L --include_feature 2>&1 | tail -2
done
for L in 1 2 3; do
  echo "renders_npy 12d level $L: $(ls output/teatime_e32_12d_$L/train/ours_None/renders_npy | wc -l)"
done

ln -sfn teatime dataset/lerf_ovs/label/teatime_e32_12d
cd eval
for MODE in "" "--prompt_ensemble"; do
  echo "--- 12d eval mode=$MODE"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name teatime_e32_12d \
      --clip_ae_ckpt ../ckpt/teatime_e32_12d/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh 0.4 --encoder_dims 256 128 64 32 12 \
      --decoder_dims 16 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion400m_e32 $MODE 2>&1 | grep -E "mIoU|mAcc"
done
cd ..
echo "==== DIM=12 done $(date +%H:%M:%S) ===="

# ---- DIM=16: full chain ----
echo "==== DIM=16 start $(date +%H:%M:%S) ===="
$PY - <<'EOF'
import re
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', '#define NUM_CHANNELS_language_feature 16'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  'torch.zeros((self._xyz.shape[0], 16), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  'torch.zeros((means3D.shape[0], 16), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read()
    s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'
    open(f, 'w').write(s2)
print('patched to 16')
EOF

cd submodules/langsplat-rasterization
rm -rf build
$PY setup.py build_ext --inplace > /tmp/compile_16d.log 2>&1 || { echo "COMPILE FAIL"; tail -30 /tmp/compile_16d.log; exit 1; }
cd ../..
echo "rasterizer compiled to 16 d"

$PY -u autoencoder/train.py --dataset_path $SRC \
    --data_subdir language_features_laion400m_e32 \
    --dataset_name teatime_e32_16d \
    --encoder_dims 256 128 64 32 16 \
    --decoder_dims 16 32 64 128 256 256 512 2>&1 | tail -3
ls ckpt/teatime_e32_16d/best_ckpt.pth

$PY -u -m mm_langsplat.encode_dim3 \
    --dataset_path $SRC \
    --data_subdir language_features_laion400m_e32 \
    --out_subdir language_features_dim16_e32 \
    --ae_ckpt ckpt/teatime_e32_16d/best_ckpt.pth \
    --encoder_dims 256 128 64 32 16 \
    --decoder_dims 16 32 64 128 256 256 512 2>&1 | tail -3
ls $SRC/language_features_dim16_e32/ | wc -l

for L in 1 2 3; do
  echo "--- 16d GS level $L start $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/teatime_e32_16d \
      --language_features_name language_features_dim16_e32 \
      --feature_level $L --include_feature \
      --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
      --port 5559$L 2>&1 | tail -2
  ls output/teatime_e32_16d_$L/chkpnt30000.pth
done

for L in 1 2 3; do
  $PY -u render.py -m output/teatime_e32_16d_$L --include_feature 2>&1 | tail -2
done
for L in 1 2 3; do
  echo "renders_npy 16d level $L: $(ls output/teatime_e32_16d_$L/train/ours_None/renders_npy | wc -l)"
done

ln -sfn teatime dataset/lerf_ovs/label/teatime_e32_16d
cd eval
for MODE in "" "--prompt_ensemble"; do
  echo "--- 16d eval mode=$MODE"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name teatime_e32_16d \
      --clip_ae_ckpt ../ckpt/teatime_e32_16d/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh 0.4 --encoder_dims 256 128 64 32 16 \
      --decoder_dims 16 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion400m_e32 $MODE 2>&1 | grep -E "mIoU|mAcc"
done
cd ..
echo "==== DIM=16 done $(date +%H:%M:%S) ===="
echo "ALL_EXP_DONE"
