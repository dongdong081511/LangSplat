#!/bin/bash
# EXP-087 (E4a teatime 补格): teatime a05/a09 训练 + 渲染 + exp086 重跑
# a05 tile 已存在 (α=0.5 重建验证通过); a09 tile 本脚本前置已生成 (α=0.9, guard 通过)
# 完成后重跑 exp086_e4bc.py 全量 → exp087 log (其他场景行应与 exp086 逐位一致, 作自洽校验)
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export LD_LIBRARY_PATH=/home/xiedexia/.conda/envs/langsplat/lib:$LD_LIBRARY_PATH
echo "==== EXP-087 teatime fill start $(date +%H:%M:%S)"

grep -q "NUM_CHANNELS_language_feature 32" submodules/langsplat-rasterization/cuda_rasterizer/config.h || { echo RASTERIZER_NOT_32D; exit 1; }

TEA=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime

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

# --- 1. teatime a05 场 (tile 在, α=0.5 已重建验证) ---
train_field $TEA teatime_dino_32ds_a05 language_features_dim32_dino_smooth_a05 6161
render_field teatime_dino_32ds_a05

# --- 2. teatime a09 场 (tile 在, α=0.9, guard 已过) ---
train_field $TEA teatime_dino_32ds_a09 language_features_dim32_dino_smooth_a09 6171
render_field teatime_dino_32ds_a09

# --- 3. 重跑 exp086 全量评测 (teatime 补 a05/a09; 其余场景行 = 自洽校验) ---
cd eval
echo "--- exp086 rerun $(date +%H:%M:%S)"
$PY -u exp086_e4bc.py 2>&1 | tee ../eval_result/exp087_e4bc_teatime_fill.log
echo "==== EXP-087 done $(date +%H:%M:%S)"
