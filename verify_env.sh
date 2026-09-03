#!/bin/bash
echo "========== LangSplat环境验证 =========="
echo ""
echo "1. 数据集目录:"
ls -la ~/project/LangSplat/dataset/ 2>/dev/null | head -20
echo ""
echo "2. Python环境:"
which python && python --version
echo ""
echo "3. PyTorch和CUDA:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "4. 关键Python包:"
pip list | grep -E "torch|segment|open_clip|clip|numpy|opencv"
echo ""
echo "5. 项目结构:"
ls -la *.py 2>/dev/null
ls -la autoencoder/ 2>/dev/null | head -10
ls -la eval/ 2>/dev/null | head -10
echo ""
echo "6. SAM checkpoints:"
ls -la ckpts/ 2>/dev/null || find . -name "sam_vit*.pth" 2>/dev/null | head -5
echo ""
echo "7. 已有训练数据:"
find ~/project/LangSplat/dataset/ -name "language_feature*" -type d 2>/dev/null | head -10
find ~/project/LangSplat/output/ -maxdepth 1 -type d 2>/dev/null | head -10
echo ""
echo "========== 验证完成 =========="
