#!/bin/bash
# Ramen场景重新训练脚本 (使用新的学习率参数)

set -e

SCENE="ramen"
PROJECT_ROOT="/home/xiedexia/project/LangSplat"

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate langsplat
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  开始训练 Ramen 场景 (language_feature_lr=0.01)"
echo "============================================================"

# 训练Level 1
echo ""
echo "=== 训练 Level 1 ==="
python train.py \
    -s "dataset/lerf_ovs/$SCENE" \
    -m "output/lerf_${SCENE}_1" \
    --include_feature \
    --iterations 20000 \
    --checkpoint_iterations 7000 15000 20000 \
    --feature_level 1 \
    --start_checkpoint "output/lerf_$SCENE/chkpnt30000.pth" \
    --eval

# 训练Level 2
echo ""
echo "=== 训练 Level 2 ==="
python train.py \
    -s "dataset/lerf_ovs/$SCENE" \
    -m "output/lerf_${SCENE}_2" \
    --include_feature \
    --iterations 20000 \
    --checkpoint_iterations 7000 15000 20000 \
    --feature_level 2 \
    --start_checkpoint "output/lerf_$SCENE/chkpnt30000.pth" \
    --eval

# 训练Level 3
echo ""
echo "=== 训练 Level 3 ==="
python train.py \
    -s "dataset/lerf_ovs/$SCENE" \
    -m "output/lerf_${SCENE}_3" \
    --include_feature \
    --iterations 20000 \
    --checkpoint_iterations 7000 15000 20000 \
    --feature_level 3 \
    --start_checkpoint "output/lerf_$SCENE/chkpnt30000.pth" \
    --eval

echo ""
echo "============================================================"
echo "  训练完成，开始渲染"
echo "============================================================"

# 渲染
python render.py -m "output/lerf_${SCENE}_1" --include_feature --feature_level 1 --iteration 20000
python render.py -m "output/lerf_${SCENE}_2" --include_feature --feature_level 2 --iteration 20000
python render.py -m "output/lerf_${SCENE}_3" --include_feature --feature_level 3 --iteration 20000

echo ""
echo "============================================================"
echo "  渲染完成，开始评估"
echo "============================================================"

# 评估
cd eval
python evaluate_iou_loc.py \
    --dataset_name "lerf_$SCENE" \
    --feat_dir "../output" \
    --ae_ckpt_dir "../autoencoder/ckpt" \
    --output_dir "../eval_result_new/$SCENE" \
    --json_folder "../dataset/lerf_ovs/label" \
    --mask_thresh 0.4

echo ""
echo "============================================================"
echo "  全部完成！"
echo "============================================================"
