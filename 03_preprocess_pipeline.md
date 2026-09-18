# MM-LangSplat 改造方案（分册 03：离线多模态特征预处理）

> **本册定位**：运行任务。调用提取器批量产出 features_mm/ 目录；含一键脚本与超参默认值。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

---

## 第三阶段：离线多模态特征预处理

### 3.1 修改 LangSplat 的 preprocess.py

**原文件**：`preprocess.py`（只提 CLIP）
**新文件**：`mm_langsplat/preprocess_mm.py`

```python
"""
多模态特征离线提取脚本。

输入：LERF / 3D-OVS 场景的多视角图像
输出：每个视角存 5 个模态特征到 .npy / .pt 文件

目录结构：
  data/LERF/ramen/
  ├── images/
  │   ├── frame_00001.jpg
  │   └── ...
  ├── features_mm/                    # 新建
  │   ├── clip/
  │   │   ├── frame_00001.pt         # [512, H, W]
  │   │   └── ...
  │   ├── dino/
  │   │   ├── frame_00001.pt         # [1024, H, W]
  │   │   └── ...
  │   ├── depth/
  │   │   ├── frame_00001.pt         # [1, H, W]
  │   │   └── ...
  │   ├── normal/
  │   │   ├── frame_00001.pt         # [3, H, W]
  │   │   └── ...
  │   ├── texture/
  │   │   ├── frame_00001.pt         # [64, H, W]
  │   │   └── ...
  │   └── masks_sam/                 # 复用 LangSplat 已有
  │       ├── frame_00001_subpart.pt
  │       ├── frame_00001_part.pt
  │       └── frame_00001_whole.pt
  └── ...
"""

import os
import torch
from tqdm import tqdm
from mm_langsplat.extractors import MultiModalExtractor

def preprocess_scene(scene_dir, output_dir, device='cuda'):
    """提取场景所有图像的多模态特征。"""
    extractor = MultiModalExtractor(
        use_clip=True,
        use_dino=True,
        use_depth=True,
        use_normal=True,
        use_texture=True,
    ).to(device)
    
    image_dir = os.path.join(scene_dir, 'images')
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # 创建输出目录
    for modality in ['clip', 'dino', 'depth', 'normal', 'texture']:
        os.makedirs(os.path.join(output_dir, modality), exist_ok=True)
    
    for img_file in tqdm(image_files, desc=f"Processing {scene_dir}"):
        # 加载图像
        image = load_image(os.path.join(image_dir, img_file))  # [1, 3, H, W]
        image = image.to(device)
        
        # 提取特征
        feats = extractor(image)
        
        # 保存
        base_name = os.path.splitext(img_file)[0]
        for modality, feat in feats.items():
            save_path = os.path.join(output_dir, modality, f'{base_name}.pt')
            torch.save(feat.cpu(), save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    preprocess_scene(args.scene_dir, args.output_dir)
```

### 3.2 预处理命令

```bash
# 对每个场景执行
python -m mm_langsplat.preprocess_mm \
    --scene_dir data/LERF/ramen \
    --output_dir data/LERF/ramen/features_mm

# 批量处理
for scene in ramen figurines teatime waldo_kitchen; do
    python -m mm_langsplat.preprocess_mm \
        --scene_dir data/LERF/$scene \
        --output_dir data/LERF/$scene/features_mm
done

# 3D-OVS 数据集
for scene in bed bench room sofa lawn; do
    python -m mm_langsplat.preprocess_mm \
        --scene_dir data/3D-OVS/$scene \
        --output_dir data/3D-OVS/$scene/features_mm
done
```

### 3.3 验收

- 每个场景的 `features_mm/` 目录下 5 个子目录都有对应文件
- 文件大小合理：CLIP ≈ 2MB/帧，DINO ≈ 4MB/帧，Depth ≈ 4KB/帧，Normal ≈ 12KB/帧，Texture ≈ 256KB/帧
- 单场景预处理时间：~30 分钟（取决于帧数）

---


---

## 第九阶段：实验执行总流程

### 9.1 一键执行脚本

**文件**：`scripts/run_full_pipeline.sh`

```bash
#!/bin/bash
# MM-LangSplat 完整训练与评测流程

set -e
SCENE=$1  # e.g. ramen
DATA_DIR=data/LERF/$SCENE
OUTPUT_DIR=output/$SCENE

mkdir -p $OUTPUT_DIR pretrained

# 阶段 0：复现 LangSplat baseline
echo "=== Stage 0: Reproduce LangSplat baseline ==="
python train.py --config arguments/lego/${SCENE}.py
python preprocess.py --source_path $DATA_DIR
python train.py --config arguments/lego/${SCENE}_langsplat.py

# 阶段 1-2：环境扩展（一次性）
echo "=== Stage 1-2: Setup multimodal extractors ==="
# 已通过 conda 与 pip 安装

# 阶段 3：多模态特征提取
echo "=== Stage 3: Extract multi-modal features ==="
python -m mm_langsplat.preprocess_mm \
    --scene_dir $DATA_DIR \
    --output_dir $DATA_DIR/features_mm

# 阶段 4：训练融合网络
echo "=== Stage 4: Train fusion network ==="
python -m mm_langsplat.trainers.train_fusion \
    --scene_dir $DATA_DIR \
    --output pretrained/fusion_${SCENE}.pt \
    --d_latent 16

# 阶段 5-6：训练语言高斯
echo "=== Stage 5-6: Train language gaussian ==="
python -m mm_langsplat.trainers.train_langsplat_mm \
    --rgb_model_path output/${SCENE}/point_cloud/iteration_30000/point_cloud.ply \
    --fusion_net_path pretrained/fusion_${SCENE}.pt \
    --model_path $OUTPUT_DIR \
    --d_latent 16 \
    --iterations 30000

# 阶段 7：评测
echo "=== Stage 7: Evaluate ==="
python -m mm_langsplat.eval_mm_langsplat \
    --model_path $OUTPUT_DIR/mm_langsplat.ply \
    --fusion_net_path pretrained/fusion_${SCENE}.pt \
    --scene_dir $DATA_DIR

echo "=== Done! ==="
```

### 9.2 完整时间预算

| 阶段 | 单场景耗时 | 备注 |
|------|-----------|------|
| 阶段 0：复现 LangSplat | 30 min | 一次 |
| 阶段 1：环境扩展 | 30 min | 一次 |
| 阶段 2：提取器实现 | 2-3 天 | 编码 + 调试 |
| 阶段 3：多模态预处理 | 30 min/场景 | 9 场景共 4.5 hr |
| 阶段 4：训练融合网络 | 30 min/场景 | 9 场景共 4.5 hr |
| 阶段 5-6：训练语言高斯 | 15 min/场景 | 9 场景共 2.25 hr |
| 阶段 7：评测 | 10 min/场景 | 9 场景共 1.5 hr |
| 阶段 8：消融实验 | 多次运行 | ~3 天 |
| **总计** | — | **~2 周（含调试）** |

---


---

## 附录 B：超参数默认值

| 超参数 | 默认值 | 备注 |
|--------|--------|------|
| `d_unified` | 256 | 模态统一维度 |
| `d_latent` | 16 | 输出 latent 维度（网格搜索对象） |
| `n_heads` | 8 | 注意力头数 |
| `n_levels` | 3 | SAM 三层级 |
| `lr_fusion` | 5e-4 | 融合网络学习率 |
| `lr_langsplat` | 5e-4 | 语言高斯学习率 |
| `epochs_fusion` | 100 | 融合网络训练轮数 |
| `iterations_langsplat` | 30000 | 语言高斯训练迭代数 |
| `batch_size` | 1 | 单视角 batch |
| `lambda_recon` | 0.1 | 重构损失权重 |
| `lambda_align` | 1.0 | CLIP 对齐损失权重 |

---

> **执行说明**：
> 1. agent 应严格按阶段 0 → 1 → 2 → ... → 8 顺序执行
> 2. 每完成一个阶段，运行对应验收命令，确认通过后再进入下一阶段
> 3. 若某阶段失败，参考第十一阶段调试


---


---
*AI生成*
