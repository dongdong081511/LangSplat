#!/bin/bash
# EXP-085 (E4a): ramen/waldo 跨场景 alpha sweep 补端点 — raw/a05/a09
# ramen: raw 需从零训 (历史清理丢失); waldo: raw ckpt 在, 仅重渲
# smooth_tiles (blend) → train ×3 levels → render; 完成后 E4(b)(c) 离线评测
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
# trae server 的 LD_LIBRARY_PATH 带旧 libstdc++ (缺 GLIBCXX_3.4.29), 前置 conda lib 修复
export LD_LIBRARY_PATH=/home/xiedexia/.conda/envs/langsplat/lib:$LD_LIBRARY_PATH
echo "==== EXP-085 E4a start $(date +%H:%M:%S)"

grep -q "NUM_CHANNELS_language_feature 32" submodules/langsplat-rasterization/cuda_rasterizer/config.h || { echo RASTERIZER_NOT_32D; exit 1; }

render_field() {  # 渲染一个场的 3 个 level (有 ckpt 才渲)
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

train_field() {  # train_field <scene_dir> <field_base> <feat_subdir> <port_base>
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

smooth_guard() {  # smooth_guard <scene_dir> <tag>: 平滑特征必须真的变了
  local SD=$1 TAG=$2
  local F1=$(ls $SD/language_features_dim32_dino/*_f.npy | head -1 | xargs basename)
  $PY -c "
import numpy as np
o = np.load('$SD/language_features_dim32_dino/$F1')
s = np.load('$SD/language_features_dim32_dino_smooth_${TAG}/$F1')
d = float(np.abs(s - o).max()); print('smooth maxdiff', d)
assert d > 1e-3, 'SMOOTH_FAILED_SILENT'
"
}

RAMEN=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/ramen
WALDO=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/waldo_kitchen

# --- 1. ramen raw 场 (缺, 从零训) ---
train_field $RAMEN ramen_dino_32d language_features_dim32_dino 6081
render_field ramen_dino_32d

# --- 2. waldo raw 场重渲 (ckpt 在) ---
render_field waldo_kitchen_dino_32d

# --- 3. smooth a05/a09 (ramen 用 ramen_dino_32d 渲染; waldo 用 waldo_kitchen_dino_32d) ---
cd eval
for SPEC in "ramen $RAMEN" "waldo_kitchen $WALDO"; do
  set -- $SPEC; SC=$1; SD=$2
  for A in 0.5 0.95; do
    TAG=$([ "$A" == "0.5" ] && echo a05 || echo a09)
    if [ ! -d $SD/language_features_dim32_dino_smooth_$TAG ]; then
      echo "--- smooth $SC $A $(date +%H:%M:%S)"
      $PY -u smooth_tiles.py --scene $SC --alpha $A --iter_tag $TAG 2>&1 | tail -1
    fi
    smooth_guard $SD $TAG
  done
done
cd ..

# --- 4. 平滑场重训 ×4 (ramen/waldo × a05/a09) ---
train_field $RAMEN ramen_dino_32ds_a05 language_features_dim32_dino_smooth_a05 6091
render_field ramen_dino_32ds_a05
train_field $RAMEN ramen_dino_32ds_a09 language_features_dim32_dino_smooth_a09 6094
render_field ramen_dino_32ds_a09
train_field $WALDO waldo_kitchen_dino_32ds_a05 language_features_dim32_dino_smooth_a05 6097
render_field waldo_kitchen_dino_32ds_a05
train_field $WALDO waldo_kitchen_dino_32ds_a09 language_features_dim32_dino_smooth_a09 6100
render_field waldo_kitchen_dino_32ds_a09

echo "==== EXP-085 E4a ALL DONE $(date +%H:%M:%S)"
