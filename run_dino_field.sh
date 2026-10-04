#!/bin/bash
# EXP-035: DINO 32d feature field full chain (teatime) + image-query eval vs CLIP 8d field
set -e -o pipefail
cd /home/xiedexia/project/LangSplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export TORCH_CUDA_ARCH_LIST="8.9"
export PATH=/home/xiedexia/.conda/envs/langsplat/bin:$PATH
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime

echo "==== DINO field start $(date +%H:%M:%S) ===="

$PY -u autoencoder/train.py --dataset_path $SRC \
    --data_subdir language_features_dino \
    --dataset_name teatime_dino_32d --input_dim 768 \
    --encoder_dims 256 128 32 32 32 \
    --decoder_dims 32 128 256 256 768 2>&1 | tail -3
ls ckpt/teatime_dino_32d/best_ckpt.pth

$PY -u -m mm_langsplat.encode_dim3 \
    --dataset_path $SRC \
    --data_subdir language_features_dino \
    --out_subdir language_features_dim32_dino \
    --ae_ckpt ckpt/teatime_dino_32d/best_ckpt.pth \
    --input_dim 768 \
    --encoder_dims 256 128 32 32 32 \
    --decoder_dims 32 128 256 256 768 2>&1 | tail -5
ls $SRC/language_features_dim32_dino/ | wc -l

$PY - <<'EOF'
import re
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', '#define NUM_CHANNELS_language_feature 32'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  'torch.zeros((self._xyz.shape[0], 32), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  'torch.zeros((means3D.shape[0], 32), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read()
    s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'
    open(f, 'w').write(s2)
print('patched to 32')
EOF

cd submodules/langsplat-rasterization
rm -rf build
$PY setup.py build_ext --inplace > /tmp/compile_32d.log 2>&1 || { echo "COMPILE FAIL"; tail -30 /tmp/compile_32d.log; exit 1; }
cd ../..
echo "rasterizer compiled to 32 d"

for L in 1 2 3; do
  echo "--- 32d GS level $L start $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/teatime_dino_32d \
      --language_features_name language_features_dim32_dino \
      --feature_level $L --include_feature \
      --start_checkpoint $SRC/output/teatime_-1/chkpnt30000.pth \
      --port 5560$L 2>&1 | tail -2
  ls output/teatime_dino_32d_$L/chkpnt30000.pth
done

for L in 1 2 3; do
  $PY -u render.py -m output/teatime_dino_32d_$L --include_feature 2>&1 | tail -2
done
for L in 1 2 3; do
  echo "renders_npy 32d level $L: $(ls output/teatime_dino_32d_$L/train/ours_None/renders_npy | wc -l)"
done

cd eval
echo "--- image-query eval DINO field $(date +%H:%M:%S)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/teatime \
    --scene_root ../dataset/lerf_ovs/teatime \
    --db_render_dirs ../output/teatime_dino_32d_1 ../output/teatime_dino_32d_2 ../output/teatime_dino_32d_3 \
    --query_subdir language_features_dim32_dino --tag DINO32d 2>&1 | grep -v Warning

echo "--- image-query eval CLIP 8d field (baseline)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/teatime \
    --scene_root ../dataset/lerf_ovs/teatime \
    --db_render_dirs ../output/teatime_8d_1 ../output/teatime_8d_2 ../output/teatime_8d_3 \
    --query_subdir language_features_dim8 --tag CLIP8d 2>&1 | grep -v Warning
cd ..
echo "==== DINO field done $(date +%H:%M:%S) ===="
echo "ALL_EXP_DONE"
