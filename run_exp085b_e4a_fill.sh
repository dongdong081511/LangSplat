#!/bin/bash
# EXP-085b (E4a 补端点): teatime raw 补训 + figurines a05 补训, 凑齐 α 曲线
# teatime: raw 从零训 (历史清理丢失); figurines: smooth a05 (raw 渲染 299×3 在) → 训练
# 完成后重跑 exp086_e4bc.py 补全 (b)(c)
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export LD_LIBRARY_PATH=/home/xiedexia/.conda/envs/langsplat/lib:$LD_LIBRARY_PATH
echo "==== EXP-085b start $(date +%H:%M:%S)"

grep -q "NUM_CHANNELS_language_feature 32" submodules/langsplat-rasterization/cuda_rasterizer/config.h || { echo RASTERIZER_NOT_32D; exit 1; }

render_field() {
  local B=$1
  for L in 1 2 3; do
    local D=output/${B}_$L
    [ -f $D/chkpnt30000.pth ] || { echo NO_CKPT $D; exit 1; }
    if [ -d "$D/train/ours_None/renders_npy" ] && [ "$(ls $D/train/ours_None/renders_npy | wc -l)" -gt 100 ]; then
      echo "skip render $D"; continue
    fi
    local AV=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
    [ "$AV" -lt 60 ] && { echo DISK_LOW ${AV}G before $D; exit 1; }
    echo "--- render $D $(date +%H:%M:%S)"
    $PY -u render.py -m $D --include_feature 2>&1 | tail -1
    local N=$(ls $D/train/ours_None/renders_npy 2>/dev/null | wc -l)
    echo "renders: $N"; [ "$N" -gt 100 ] || { echo RENDER_FAIL $D; exit 1; }
  done
}

train_field() {
  local SD=$1 B=$2 FEAT=$3 PB=$4
  for L in 1 2 3; do
    [ -f output/${B}_$L/chkpnt30000.pth ] && { echo "skip train ${B}_$L"; continue; }
    echo "--- train ${B}_$L $(date +%H:%M:%S)"
    $PY -u train.py -s $SD -m output/$B \
        --language_features_name $FEAT --feature_level $L --include_feature \
        --start_checkpoint $SD/output/${SD##*/}_-1/chkpnt30000.pth \
        --port $((PB+L)) 2>&1 | tail -2
    ls output/${B}_$L/chkpnt30000.pth
  done
}

TEATIME=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime
FIGURINES=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/figurines

# --- 1. teatime raw ---
train_field $TEATIME teatime_dino_32d language_features_dim32_dino 6111
render_field teatime_dino_32d

# --- 2. figurines a05 ---
cd eval
if [ ! -d $FIGURINES/language_features_dim32_dino_smooth_a05 ]; then
  echo "--- smooth figurines 0.5 $(date +%H:%M:%S)"
  $PY -u smooth_tiles.py --scene figurines --alpha 0.5 --iter_tag a05 2>&1 | tail -1
fi
F1=$(ls $FIGURINES/language_features_dim32_dino/*_f.npy | sed -n '1p' | xargs basename)
$PY -c "
import numpy as np
o = np.load('$FIGURINES/language_features_dim32_dino/$F1')
s = np.load('$FIGURINES/language_features_dim32_dino_smooth_a05/$F1')
d = float(np.abs(s - o).max()); print('smooth maxdiff', d)
assert d > 1e-3, 'SMOOTH_FAILED_SILENT'
"
cd ..
train_field $FIGURINES figurines_dino_32ds_a05 language_features_dim32_dino_smooth_a05 6121
render_field figurines_dino_32ds_a05

echo "==== EXP-085b ALL DONE $(date +%H:%M:%S)"
