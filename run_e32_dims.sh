#!/bin/bash
# EXP-031/032 v2: lr fixed to default 1e-4, idempotent patch via regex
set -e -o pipefail
cd /home/xiedexia/project/LangSplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export TORCH_CUDA_ARCH_LIST="8.9"
export PATH=/home/xiedexia/.conda/envs/langsplat/bin:$PATH
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime

for DIM in 12 16; do
  echo "==================== DIM=$DIM start $(date +%H:%M:%S) ===================="

  # 1. patch 3 hardcoded dims -> DIM (regex, idempotent)
  $PY - <<EOF
import re
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', '#define NUM_CHANNELS_language_feature $DIM'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  'torch.zeros((self._xyz.shape[0], $DIM), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  'torch.zeros((means3D.shape[0], $DIM), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read()
    s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'
    open(f, 'w').write(s2)
print('patched to $DIM')
EOF

  # 2. recompile rasterizer
  cd submodules/langsplat-rasterization
  rm -rf build
  $PY setup.py build_ext --inplace > /tmp/compile_${DIM}d.log 2>&1 || { echo "COMPILE FAIL"; tail -30 /tmp/compile_${DIM}d.log; exit 1; }
  cd ../..
  echo "rasterizer compiled to $DIM d"

  # 3. AE training (default lr 1e-4, matching all historical AE runs)
  $PY -u autoencoder/train.py --dataset_path $SRC \
      --data_subdir language_features_laion400m_e32 \
      --dataset_name teatime_e32_${DIM}d \
      --encoder_dims 256 128 64 32 $DIM \
      --decoder_dims 16 32 64 128 256 256 512 2>&1 | tail -3
  ls ckpt/teatime_e32_${DIM}d/best_ckpt.pth

  # 4. encode 512 -> DIM
  $PY -u -m mm_langsplat.encode_dim3 \
      --dataset_path $SRC \
      --data_subdir language_features_laion400m_e32 \
      --out_subdir language_features_dim${DIM}_e32 \
      --ae_ckpt ckpt/teatime_e32_${DIM}d/best_ckpt.pth \
      --encoder_dims 256 128 64 32 $DIM \
      --decoder_dims 16 32 64 128 256 256 512 2>&1 | tail -3
  ls $SRC/language_features_dim${DIM}_e32/ | wc -l

  # 5. GS training x3 (warm start from RGB-only ckpt, len=12)
  for L in 1 2 3; do
    echo "--- GS level $L start $(date +%H:%M:%S)"
    $PY -u train.py -s $SRC -m output/teatime_e32_${DIM}d_${L} \
        --language_features_name language_features_dim${DIM}_e32 \
        --feature_level $L --include_feature \
        --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
        --port 5559$L 2>&1 | tail -2
    ls output/teatime_e32_${DIM}d_${L}/chkpnt30000.pth
  done

  # 6. render x3
  for L in 1 2 3; do
    $PY -u render.py -m output/teatime_e32_${DIM}d_${L} --include_feature 2>&1 | tail -2
  done
  for L in 1 2 3; do
    n=$(ls output/teatime_e32_${DIM}d_${L}/train/ours_None/renders_npy | wc -l)
    echo "renders_npy level $L: $n"
  done

  # 7. eval x2 (teacher-matched text encoder!)
  ln -sfn teatime dataset/lerf_ovs/label/teatime_e32_${DIM}d
  cd eval
  echo "--- eval no-template"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name teatime_e32_${DIM}d \
      --clip_ae_ckpt ../ckpt/teatime_e32_${DIM}d/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh 0.4 --encoder_dims 256 128 64 32 $DIM \
      --decoder_dims 16 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion400m_e32 2>&1 | grep -E "mIoU|mAcc"
  echo "--- eval +ensemble"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name teatime_e32_${DIM}d \
      --clip_ae_ckpt ../ckpt/teatime_e32_${DIM}d/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh 0.4 --encoder_dims 256 128 64 32 $DIM \
      --decoder_dims 16 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion400m_e32 --prompt_ensemble 2>&1 | grep -E "mIoU|mAcc"
  cd ..
  echo "==================== DIM=$DIM done $(date +%H:%M:%S) ===================="
done
echo "ALL_EXP_DONE"
