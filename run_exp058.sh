#!/bin/bash
# EXP-058: waldo DINOv2 native 分辨率救援全链路 (2D 入口已翻绿 59.62% vs CLIP 44.23%)
# A 全帧 native 特征 → B AE 32d → C encode → D patch 32d+编译 → E 3D训练×3 → F 渲染 → G image eval + McNemar
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
SRC=dataset/lerf_ovs/waldo_kitchen
BASE=$SRC/output/waldo_kitchen_-1/chkpnt30000.pth
echo "==== EXP-058 start $(date +%H:%M:%S)"

# A. 全帧 native 特征
N=$(ls $SRC/language_features_dino_native/*_f.npy 2>/dev/null | wc -l)
if [ "$N" -lt 180 ]; then
  echo "--- A native extract $(date +%H:%M:%S)"
  $PY -u eval/extract_native_all.py 2>&1 | tail -3
fi
N=$(ls $SRC/language_features_dino_native/*_f.npy | wc -l)
echo "A. native feats: $N"; [ "$N" -ge 180 ] || { echo A_FAIL; exit 1; }

# B. AE 32d (native)
if [ ! -f ckpt/waldo_kitchen_dino32n/best_ckpt.pth ]; then
  echo "--- B AE 32d native $(date +%H:%M:%S)"
  $PY -u autoencoder/train.py --dataset_path $SRC --data_subdir language_features_dino_native \
      --dataset_name waldo_kitchen_dino32n --input_dim 768 \
      --encoder_dims 256 128 32 32 32 --decoder_dims 32 128 256 256 768 2>&1 | tail -3
fi
ls ckpt/waldo_kitchen_dino32n/best_ckpt.pth

# C. encode dim32
if [ ! -d $SRC/language_features_dim32_dino_native ]; then
  echo "--- C encode $(date +%H:%M:%S)"
  $PY -u -m mm_langsplat.encode_dim3 --dataset_path $SRC --data_subdir language_features_dino_native \
      --out_subdir language_features_dim32_dino_native --ae_ckpt ckpt/waldo_kitchen_dino32n/best_ckpt.pth \
      --input_dim 768 --encoder_dims 256 128 32 32 32 --decoder_dims 32 128 256 256 768 2>&1 | tail -3
fi
ls $SRC/language_features_dim32_dino_native | wc -l

# D. patch 32d + compile
CUR=$(grep -oP 'NUM_CHANNELS_language_feature \K\d+' submodules/langsplat-rasterization/cuda_rasterizer/config.h)
if [ "$CUR" != "32" ]; then
  echo "--- D patch 32d + compile (cur=$CUR) $(date +%H:%M:%S)"
  $PY - <<'PYEOF'
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
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print('patched to 32')
PYEOF
  (cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_058.log 2>&1) || { echo COMPILE_FAIL; tail -20 /tmp/compile_058.log; exit 1; }
  echo "compiled 32d $(date +%H:%M:%S)"
else
  echo "D. skip compile (already 32d)"
fi

# E. 3D 训练 ×3 (raw, 不平滑 — waldo 前置判据先看 raw)
for L in 1 2 3; do
  [ -f output/waldo_kitchen_dino_32dn_$L/chkpnt30000.pth ] && { echo "skip E L$L"; continue; }
  echo "--- E 3D train level $L $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/waldo_kitchen_dino_32dn \
      --language_features_name language_features_dim32_dino_native --feature_level $L --include_feature \
      --start_checkpoint $BASE --port 618$L 2>&1 | tail -2
  ls output/waldo_kitchen_dino_32dn_$L/chkpnt30000.pth
done

# F. 渲染 ×3
for L in 1 2 3; do
  D=output/waldo_kitchen_dino_32dn_$L
  [ -d $D/train/ours_None/renders_npy ] && [ "$(ls $D/train/ours_None/renders_npy | wc -l)" -gt 100 ] && { echo "skip F L$L"; continue; }
  echo "--- F render level $L $(date +%H:%M:%S)"
  $PY -u render.py -m $D --include_feature 2>&1 | tail -1
done

# G. image eval + McNemar vs CLIP 场
cd eval
echo "--- G eval $(date +%H:%M:%S)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/waldo_kitchen \
  --scene_root ../dataset/lerf_ovs/waldo_kitchen \
  --db_render_dirs ../output/waldo_kitchen_dino_32dn_1 ../output/waldo_kitchen_dino_32dn_2 ../output/waldo_kitchen_dino_32dn_3 \
  --query_subdir language_features_dim32_dino_native --tag WALDO_DINO_native \
  --out_json ../eval_result/mcnemar/WALDO_DINO_native.json 2>&1 | grep -E "^\[" || echo EVAL_WARN
for P in any first majority; do
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/WALDO_CLIP8d.json \
    --dino_json ../eval_result/mcnemar/WALDO_DINO_native.json --protocol $P --tag WALDO_native_$P 2>&1 | tail -2
done
cd ..
echo "==== EXP-058 ALL DONE $(date +%H:%M:%S)"
