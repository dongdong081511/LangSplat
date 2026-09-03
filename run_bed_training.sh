#!/bin/bash
set -e

cd /home/xiedexia/project/LangSplat

echo "=== 开始bed场景训练 ==="

# 检查数据
echo "1. 检查数据..."
if [ ! -d "output/bed_data/images" ]; then
    echo "错误: bed_data目录不存在"
    exit 1
fi

# 检查AE checkpoint
if [ ! -f "autoencoder/ckpt/bed/ae_ckpt/best_ckpt.pth" ]; then
    echo "错误: AE checkpoint不存在"
    exit 1
fi

# 清理旧输出
echo "2. 清理旧输出..."
rm -rf output/bed_output

# 训练
echo "3. 开始训练..."
python train.py -s output/bed_data -m output/bed_output --eval

# 渲染
echo "4. 渲染..."
python render.py -m output/bed_output

# 评估
echo "5. 评估..."
python evaluate.py -s output/bed_data -m output/bed_output

echo "=== 完成 ==="
