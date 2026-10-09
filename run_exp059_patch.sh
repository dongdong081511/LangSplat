#!/bin/bash
# EXP-059: 64d rasterizer 改造 — patch 三处 + 重编译 + 冒烟验证
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
echo "==== EXP-059 patch+compile start $(date +%H:%M:%S)"

$PY - <<'PYEOF'
import re
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', '#define NUM_CHANNELS_language_feature 64'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  'torch.zeros((self._xyz.shape[0], 64), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  'torch.zeros((means3D.shape[0], 64), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print('patched to 64d')
PYEOF
grep -n "NUM_CHANNELS_language_feature 64" submodules/langsplat-rasterization/cuda_rasterizer/config.h
grep -n "64), device" scene/gaussian_model.py | head -1

(cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_059.log 2>&1) || { echo COMPILE_FAIL; tail -25 /tmp/compile_059.log; exit 1; }
echo "compiled 64d OK $(date +%H:%M:%S)"

# 冒烟: 100s 训练验证 forward+backward 64d 全链
timeout 120 $PY -u train.py -s dataset/lerf_ovs/waldo_kitchen -m output/smoke_64d \
    --language_features_name language_features_dim32_dino_native --feature_level 1 --include_feature \
    --start_checkpoint dataset/lerf_ovs/waldo_kitchen/output/waldo_kitchen_-1/chkpnt30000.pth \
    --port 6199 2>&1 | tail -3 || true
echo "==== smoke done $(date +%H:%M:%S) — 若上方无 CUDA error 即通过"
