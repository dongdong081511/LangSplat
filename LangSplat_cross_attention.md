# LangSplat 多模态 Cross-Attention 改造方案（Agent 执行版）

> **文档目的**：作为 agent 的执行手册，将 LangSplat 单模态框架改造为 5 模态 cross-attention 融合的 MM-LangSplat
> **执行方式**：按 8 个阶段顺序执行，每阶段都有明确的输入/输出/验收标准
> **基准代码库**：https://github.com/minghanqin/LangSplat
> **执行日期**：2026-09-07

---

## 第零阶段：环境与基准复现（先跑通原版）

### 0.1 基础环境

```bash
# 克隆 LangSplat 官方仓库
git clone https://github.com/minghanqin/LangSplat.git
cd LangSplat
git submodule update --init --recursive

# 创建 conda 环境
conda create -n mm_langsplat python=3.8 -y
conda activate mm_langsplat

# 安装 PyTorch（与原版一致）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 安装 3D-GS 子模块依赖
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

# 安装 LangSplat 原版 Python 依赖
pip install -r requirements.txt
```

### 0.2 数据集准备

下载 LERF 与 3D-OVS 数据集：
- LERF Dataset: https://github.com/lerftoy/lerf
- 3D-OVS Dataset: https://github.com/Kunhao-Liu/3D-OVS

```bash
# 数据集组织结构
data/
├── LERF/
│   ├── ramen/
│   ├── figurines/
│   ├── teatime/
│   └── waldo_kitchen/
└── 3D-OVS/
    ├── bed/
    ├── bench/
    ├── room/
    ├── sofa/
    └── lawn/
```

### 0.3 基准复现（验收标准：mIoU 51.4%）

按 LangSplat README 跑通：
1. 训练 RGB 场景：`python train.py --config arguments/lego/lerf_ramen.py`
2. 提取 CLIP 特征：`python preprocess.py --source_path data/LERF/ramen/...`
3. 训练语言高斯：`python train.py --config arguments/lego/lerf_ramen_langsplat.py`
4. 评测：`python eval_langsplat.py`

**验收**：在 LERF 上 mIoU ≈ 51.4%，作为后续改进的 baseline。

---

## 第一阶段：环境扩展与多模态依赖安装

### 1.1 各模态模型信息汇总表（agent 速查）

| 模态 | 模型 | 引用论文 | 论文链接 | GitHub 仓库 | 安装方式 |
|------|------|---------|---------|------------|---------|
| CLIP（语义） | OpenCLIP ViT-B/16 | Radford et al., ICML 2021, "Learning Transferable Visual Models From Natural Language Supervision" | https://arxiv.org/abs/2103.00020 | https://github.com/openai/CLIP | `pip install open_clip_torch` |
| DINOv2（视觉） | DINOv2 ViT-L/14 | Oquab et al., TMLR 2024, "DINOv2: Learning Robust Visual Features without Supervision" | https://arxiv.org/abs/2304.07193 | https://github.com/facebookresearch/dinov2 | `pip install -e .` 或 torch.hub |
| Depth（深度） | Depth Anything V2（ViT-L） | Yang et al., NeurIPS 2024, "Depth Anything V2" | https://arxiv.org/abs/2406.09414 | https://github.com/DepthAnything/Depth-Anything-V2 | 下载权重，直接 import |
| Normal（法向量） | DSINE | Bae et al., ECCV 2024, "DSINE: Revisiting Depth-aware Normal Estimation" | https://arxiv.org/abs/2403.18205 | https://github.com/baegwangbin/DSINE | clone + 加载权重 |
| Texture（纹理） | STEGO | Hamilton et al., ICLR 2022, "Unsupervised Semantic Segmentation by Distilling Feature Correspondences" | https://arxiv.org/abs/2207.05026 | https://github.com/hamarb172/STEGO | clone + install |
| SAM（分割） | SAM ViT-H | Kirillov et al., ICCV 2023, "Segment Anything" | https://arxiv.org/abs/2304.02643 | https://github.com/facebookresearch/segment-anything | `pip install git+https://github.com/facebookresearch/segment-anything.git` |

### 1.2 安装多模态依赖

```bash
# OpenCLIP（CLIP 实现，与 LangSplat 一致）
pip install open_clip_torch==2.24.0

# DINOv2（通过 torch.hub 加载，无需 clone）
# 已包含在 torch.hub 中，运行时自动下载

# Depth Anything V2
cd ${PROJECT_ROOT}/third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt
# 下载预训练权重
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
cd ${PROJECT_ROOT}

# DSINE（法向量）
cd ${PROJECT_ROOT}/third_party
git clone https://github.com/baegwangbin/DSINE.git
# 下载权重
cd DSINE
wget https://huggingface.co/baegwangbin/DSINE/resolve/main/dsine.pt
cd ${PROJECT_ROOT}

# STEGO（纹理）
cd ${PROJECT_ROOT}/third_party
git clone https://github.com/hamarb172/STEGO.git
cd STEGO
pip install -r requirements.txt
# 下载 ImageNet 预训练权重
mkdir -p logs && cd logs
wget https://huggingface.co/hamarb172/STEGO/resolve/main/redirected_latest.pt
cd ${PROJECT_ROOT}

# SAM（LangSplat 已有，确保安装）
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 1.3 第三方依赖目录结构（推荐）

```
LangSplat/
├── third_party/
│   ├── Depth-Anything-V2/
│   ├── DSINE/
│   └── STEGO/
├── pretrained/
│   ├── depth_anything_v2_vitl.pth
│   ├── dsine.pt
│   ├── redirect_latest.pt
│   ├── sam_vit_h_4b8939.pth
│   └── dinov2_vitl14_pretrain.pth
├── mm_langsplat/                   # 新建：多模态融合模块
│   ├── __init__.py
│   ├── extractors/                # 各模态提取器
│   │   ├── __init__.py
│   │   ├── clip_extractor.py
│   │   ├── dino_extractor.py
│   │   ├── depth_extractor.py
│   │   ├── normal_extractor.py
│   │   ├── texture_extractor.py
│   │   └── sam_extractor.py
│   ├── fusion/                    # 融合网络
│   │   ├── __init__.py
│   │   ├── cross_attention.py     # 主推方案
│   │   ├── concat_mlp.py          # baseline
│   │   ├── transformer_encoder.py # baseline
│   │   └── moe.py                 # baseline
│   ├── models/
│   │   ├── __init__.py
│   │   └── mm_language_gaussian.py  # 多模态语言高斯模型
│   ├── data/
│   │   ├── __init__.py
│   │   └── multimodal_dataset.py
│   └── trainers/
│       ├── __init__.py
│       ├── train_fusion.py        # 训练融合网络
│       └── train_langsplat_mm.py  # 训练语言高斯
└── ... (LangSplat 原有文件)
```

**验收**：
- 所有 `import` 可正常加载
- 所有预训练权重文件存在并 sha256 校验通过
- DINOv2 `torch.hub.load(...)` 可正常加载

---

## 第二阶段：多模态特征提取器实现

### 2.1 总体规范

每个提取器统一接口：
- 输入：`image: torch.Tensor`，形状 `[B, 3, H, W]`，范围 `[0, 1]`
- 输出：`dict[str, torch.Tensor]`，含 `feature`、`mask`（如有）等

### 2.2 模态 1：CLIP 提取器（复用 LangSplat，做适配）

**文件**：`mm_langsplat/extractors/clip_extractor.py`

```python
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPExtractor(nn.Module):
    """OpenCLIP ViT-B/16，输出像素级 CLIP 特征（用 SAM mask 聚合）。
    
    Reference:
        Radford et al., "Learning Transferable Visual Models From Natural 
        Language Supervision", ICML 2021.
        https://arxiv.org/abs/2103.00020
        https://github.com/openai/CLIP
    """
    def __init__(self, model_name='ViT-B-16', pretrained='openai'):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 512
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor, masks: list = None) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W], range [0, 1]
            masks: 可选，来自 SAM 三层级 mask
        Returns:
            features: [B, 512, H, W] 像素级 CLIP 特征
        """
        # 如果提供 mask，按 LangSplat 公式 (1) 提取
        if masks is not None:
            return self._extract_with_masks(image, masks)
        # 否则直接整图编码 + 上采样
        feat = self.model.encode_image(image)  # [B, 512]
        feat = feat.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        feat = F.interpolate(feat, size=image.shape[-2:], mode='bilinear')
        return feat
    
    def _extract_with_masks(self, image, masks):
        # 沿用 LangSplat preprocess.py 中的 mask 聚合逻辑
        # 见 LangSplat preprocess.py 第 100-150 行
        pass
```

**对应 LangSplat 原文件**：`preprocess.py` 中的 `clip_extract` 函数（直接复用）。

### 2.3 模态 2：DINOv2 提取器

**文件**：`mm_langsplat/extractors/dino_extractor.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Extractor(nn.Module):
    """DINOv2 ViT-L/14，输出像素级视觉特征。
    
    Reference:
        Oquab et al., "DINOv2: Learning Robust Visual Features without 
        Supervision", TMLR 2024.
        https://arxiv.org/abs/2304.07193
        https://github.com/facebookresearch/dinov2
    
    Args:
        model_name: 'dinov2_vits14' (384d) / 'dinov2_vitb14' (768d) / 'dinov2_vitl14' (1024d)
                    推荐使用 vitb14 平衡精度与显存
    """
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch_size = 14
        self.dim_out = self.model.embed_dim  # vitl14: 1024; vitb14: 768
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W], range [0, 1]
        Returns:
            features: [B, dim_out, H, W] 像素级特征（已上采样）
        """
        B, _, H, W = image.shape
        # DINOv2 要求 H, W 是 patch_size 的倍数
        H_pad = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_pad = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        image = F.interpolate(image, size=(H_pad, W_pad), mode='bilinear')
        
        # 提取 patch-level 特征
        # 输出 shape: [B, num_patches, dim_out]
        # num_patches = (H_pad/14) * (W_pad/14)
        feats = self.model.forward_features(image)  # [B, N, D]
        
        # 去掉 CLS token
        if feats[:, 0].mean() != feats[:, 1].mean():  # 简单判断
            feats = feats[:, 1:]
        
        # Reshape 到 2D 网格
        h_grid = H_pad // self.patch_size
        w_grid = W_pad // self.patch_size
        feats = feats.transpose(1, 2).reshape(B, -1, h_grid, w_grid)
        # 上采样到原图尺寸
        feats = F.interpolate(feats, size=(H, W), mode='bilinear')
        return feats  # [B, D, H, W]
```

**关键点**：
- DINOv2 输入图像必须 resize 到 14 的倍数
- 输出是 patch-level，需上采样到像素级
- 选 `vitl14`（1024 维）精度最高；选 `vitb14`（768 维）显存更省

### 2.4 模态 3：Depth Anything V2 提取器

**文件**：`mm_langsplat/extractors/depth_extractor.py`

```python
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加入第三方路径
sys.path.append('third_party/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

class DepthExtractor(nn.Module):
    """Depth Anything V2，输出相对深度图。
    
    Reference:
        Yang et al., "Depth Anything V2", NeurIPS 2024.
        https://arxiv.org/abs/2406.09414
        https://github.com/DepthAnything/Depth-Anything-V2
    """
    def __init__(self, model_path='pretrained/depth_anything_v2_vitl.pth',
                 model_type='vitl'):
        super().__init__()
        self.model = DepthAnythingV2(
            encoder=model_type,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 1
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W], range [0, 1]
        Returns:
            depth: [B, 1, H, W] 归一化深度（0=近，1=远）
        """
        # Depth Anything V2 输入要求 [0, 1] 的 RGB
        depth = self.model.infer_image(image)  # [B, H, W]
        # 归一化到 [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        return depth.unsqueeze(1)  # [B, 1, H, W]
```

**关键点**：
- 输出是相对深度（无真实尺度），需归一化
- LERF 数据集有真实相机位姿，理论上可用 COLMAP 算真实深度；这里先用预测深度保持简单

### 2.5 模态 4：DSINE 法向量提取器

**文件**：`mm_langsplat/extractors/normal_extractor.py`

```python
import sys
import torch
import torch.nn as nn

sys.path.append('third_party/DSINE')
from dsine.models import DSINEModel

class NormalExtractor(nn.Module):
    """DSINE 法向量预测，输出相机坐标系下的法向量。
    
    Reference:
        Bae et al., "DSINE: Revisiting Depth-aware Normal Estimation", 
        ECCV 2024.
        https://arxiv.org/abs/2403.18205
        https://github.com/baegwangbin/DSINE
    """
    def __init__(self, model_path='pretrained/dsine.pt'):
        super().__init__()
        self.model = DSINEModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 3
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor, intrinsics=None) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W], range [0, 1]
            intrinsics: 可选，相机内参 [B, 3, 3]
        Returns:
            normal: [B, 3, H, W] 相机坐标系法向量
        """
        # DSINE 需要归一化输入到 ImageNet 标准
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        if intrinsics is None:
            # 默认内参
            B, _, H, W = image.shape
            intrinsics = torch.tensor([
                [W/2, 0, W/2],
                [0, H/2, H/2],
                [0, 0, 1]
            ]).unsqueeze(0).repeat(B, 1, 1).to(image.device)
        
        normal = self.model(image, intrinsics=intrinsics)  # [B, 3, H, W]
        # 归一化到单位向量
        normal = F.normalize(normal, dim=1)
        return normal
```

### 2.6 模态 5：STEGO 纹理提取器

**文件**：`mm_langsplat/extractors/texture_extractor.py`

```python
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('third_party/STEGO')
from stego import STEGOModel

class TextureExtractor(nn.Module):
    """STEGO 无监督纹理分割，输出纹理特征。
    
    Reference:
        Hamilton et al., "Unsupervised Semantic Segmentation by 
        Distilling Feature Correspondences", ICLR 2022.
        https://arxiv.org/abs/2207.05026
        https://github.com/hamarb172/STEGO
    """
    def __init__(self, ckpt_path='third_party/STEGO/logs/redirected_latest.pt'):
        super().__init__()
        self.model = STEGOModel.load_from_checkpoint(ckpt_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 64  # STEGO 中间层特征维度
    
    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W], range [0, 1]
        Returns:
            texture: [B, 64, H, W] 纹理特征
        """
        # STEGO 内部 forward 提取中间特征
        feats = self.model.forward_features(image)  # [B, 64, H//4, W//4]
        feats = F.interpolate(feats, size=image.shape[-2:], mode='bilinear')
        return feats
```

### 2.7 SAM 三层级分割（复用 LangSplat 已有）

**文件**：`mm_langsplat/extractors/sam_extractor.py`

> LangSplat 已有 SAM 集成，直接复用 `preprocess.py` 中的逻辑。三层级输出 `M_s, M_p, M_w`。

```python
# 直接复用 LangSplat 的 preprocess.py 中的 SAM 调用
# 输入：图像
# 输出：3 个 mask（subpart / part / whole）
# 详见 LangSplat preprocess.py
```

### 2.8 统一多模态提取器封装

**文件**：`mm_langsplat/extractors/__init__.py`

```python
import torch
import torch.nn as nn
from .clip_extractor import CLIPExtractor
from .dino_extractor import DINOv2Extractor
from .depth_extractor import DepthExtractor
from .normal_extractor import NormalExtractor
from .texture_extractor import TextureExtractor

class MultiModalExtractor(nn.Module):
    """统一封装 5 模态特征提取。
    
    输出字典包含 5 个键，每个对应一个模态的像素级特征。
    """
    def __init__(self, 
                 use_clip=True, 
                 use_dino=True, 
                 use_depth=True,
                 use_normal=True, 
                 use_texture=True):
        super().__init__()
        self.use_clip = use_clip
        self.use_dino = use_dino
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.use_texture = use_texture
        
        if use_clip:
            self.clip = CLIPExtractor()
        if use_dino:
            self.dino = DINOv2Extractor(model_name='dinov2_vitl14')
        if use_depth:
            self.depth = DepthExtractor()
        if use_normal:
            self.normal = NormalExtractor()
        if use_texture:
            self.texture = TextureExtractor()
    
    @torch.no_grad()
    def forward(self, image, masks=None):
        """
        Args:
            image: [B, 3, H, W]
            masks: 来自 SAM 的三层级 mask
        Returns:
            dict with keys: 'clip', 'dino', 'depth', 'normal', 'texture'
            每个值是 [B, D_m, H, W] 张量
        """
        feats = {}
        if self.use_clip:
            feats['clip'] = self.clip(image, masks)
        if self.use_dino:
            feats['dino'] = self.dino(image)
        if self.use_depth:
            feats['depth'] = self.depth(image)
        if self.use_normal:
            feats['normal'] = self.normal(image)
        if self.use_texture:
            feats['texture'] = self.texture(image)
        return feats
    
    @property
    def dims(self):
        """返回各模态输出维度字典"""
        return {
            'clip': 512 if self.use_clip else 0,
            'dino': 1024 if self.use_dino else 0,  # vitl14
            'depth': 1 if self.use_depth else 0,
            'normal': 3 if self.use_normal else 0,
            'texture': 64 if self.use_texture else 0,
        }
```

**验收**：跑通 `python -c "from mm_langsplat.extractors import MultiModalExtractor; ext = MultiModalExtractor(); print(ext.dims)"`，输出各模态维度。

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

## 第四阶段：Cross-Attention 融合网络实现

### 4.1 主方案：Hierarchical Cross-modal Fusion（HCF）

**文件**：`mm_langsplat/fusion/cross_attention.py`

```python
"""
跨模态交叉注意力融合网络（Cross-Modal Fusion Network, CMFN）。

输入：5 模态像素级特征 dict
    - 'clip':    [B, 512, H, W]
    - 'dino':    [B, 1024, H, W]  (or 768 for vitb14)
    - 'depth':   [B, 1, H, W]
    - 'normal':  [B, 3, H, W]
    - 'texture': [B, 64, H, W]

输出：融合后的压缩 latent [B, d_latent, H, W]，d_latent 为超参（推荐 16）

架构：
    1. 各模态线性投影到统一维度 D_u (256)
    2. 模态内 self-attention（在 H*W 维度上做）
    3. 跨模态 cross-attention（CLIP 作为 query，其他模态作为 K/V）
    4. 模态门控（动态加权各模态）
    5. 投影到 d_latent 维
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjector(nn.Module):
    """单模态线性投影 + LayerNorm"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
    
    def forward(self, x):
        # x: [B, D_in, H, W] -> [B, H, W, D_out]
        x = x.permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.norm(x)
        return x  # [B, H, W, D_out]


class CrossModalFusionNetwork(nn.Module):
    """主推方案：跨模态交叉注意力融合。
    
    Args:
        dim_clip: CLIP 特征维度，默认 512
        dim_dino: DINOv2 特征维度，默认 1024 (vitl14) 或 768 (vitb14)
        dim_depth: 深度维度，固定 1
        dim_normal: 法向量维度，固定 3
        dim_texture: 纹理维度，默认 64
        d_unified: 统一中间维度，默认 256
        d_latent: 输出 latent 维度，推荐 16（网格搜索）
        n_heads: 注意力头数，默认 8
        use_normal: 是否启用法向量模态
        use_texture: 是否启用纹理模态
    """
    def __init__(self,
                 dim_clip=512,
                 dim_dino=1024,
                 dim_depth=1,
                 dim_normal=3,
                 dim_texture=64,
                 d_unified=256,
                 d_latent=16,
                 n_heads=8,
                 use_normal=True,
                 use_texture=True):
        super().__init__()
        self.use_normal = use_normal
        self.use_texture = use_texture
        self.d_unified = d_unified
        self.d_latent = d_latent
        
        # 步骤 1：模态投影
        self.proj_clip = ModalityProjector(dim_clip, d_unified)
        self.proj_dino = ModalityProjector(dim_dino, d_unified)
        self.proj_depth = ModalityProjector(dim_depth, d_unified)
        if use_normal:
            self.proj_normal = ModalityProjector(dim_normal, d_unified)
        if use_texture:
            self.proj_texture = ModalityProjector(dim_texture, d_unified)
        
        # 步骤 2：模态内 self-attention
        # 在 H*W 像素维度上做 attention，token 维度是 D_u
        self.self_attn_clip = nn.MultiheadAttention(d_unified, n_heads, batch_first=True)
        self.self_attn_dino = nn.MultiheadAttention(d_unified, n_heads, batch_first=True)
        self.self_attn_depth = nn.MultiheadAttention(d_unified, n_heads, batch_first=True)
        if use_normal:
            self.self_attn_normal = nn.MultiheadAttention(d_unified, n_heads, batch_first=True)
        if use_texture:
            self.self_attn_texture = nn.MultiheadAttention(d_unified, n_heads, batch_first=True)
        
        # 步骤 3：跨模态 cross-attention
        # CLIP 作为 query，其他模态作为 K/V
        n_kv_mods = 3 + int(use_normal) + int(use_texture)
        self.cross_attn = nn.MultiheadAttention(d_unified, n_heads, batch_first=True)
        # 用可学习的 modality token 区分 K/V 来源
        self.modality_tokens = nn.Parameter(torch.randn(n_kv_mods, d_unified) * 0.02)
        
        # 步骤 4：模态门控
        n_mods = 2 + n_kv_mods  # clip + dino + kv mods
        self.gate = nn.Sequential(
            nn.Linear(d_unified, d_unified // 2),
            nn.GELU(),
            nn.Linear(d_unified // 2, n_mods),
        )
        
        # 步骤 5：投影压缩
        self.proj_out = nn.Sequential(
            nn.Linear(d_unified, d_unified),
            nn.GELU(),
            nn.Linear(d_unified, d_latent),
        )
    
    def forward(self, feats):
        """
        Args:
            feats: dict with keys 'clip', 'dino', 'depth', 
                   optional 'normal', 'texture'
                   each value: [B, D_m, H, W]
        Returns:
            fused_latent: [B, d_latent, H, W]
            gate_weights: [B, H, W, n_mods] 用于可视化
        """
        B, _, H, W = feats['clip'].shape
        
        # 步骤 1：投影到统一维度
        h_clip = self.proj_clip(feats['clip'])        # [B, H, W, D_u]
        h_dino = self.proj_dino(feats['dino'])
        h_depth = self.proj_depth(feats['depth'])
        h_kv_list = [h_dino, h_depth]
        if self.use_normal:
            h_normal = self.proj_normal(feats['normal'])
            h_kv_list.append(h_normal)
        if self.use_texture:
            h_texture = self.proj_texture(feats['texture'])
            h_kv_list.append(h_texture)
        
        # 展平空间维度用于 attention
        # [B, H, W, D_u] -> [B, H*W, D_u]
        def flatten(x):
            return x.reshape(B, H * W, -1)
        
        h_clip_flat = flatten(h_clip)
        h_kv_flat = [flatten(h) for h in h_kv_list]
        
        # 步骤 2：模态内 self-attention
        h_clip_sa, _ = self.self_attn_clip(h_clip_flat, h_clip_flat, h_clip_flat)
        h_dino_sa, _ = self.self_attn_dino(h_kv_flat[0], h_kv_flat[0], h_kv_flat[0])
        h_depth_sa, _ = self.self_attn_depth(h_kv_flat[1], h_kv_flat[1], h_kv_flat[1])
        h_kv_sa_list = [h_dino_sa, h_depth_sa]
        if self.use_normal:
            h_normal_sa, _ = self.self_attn_normal(h_kv_flat[2], h_kv_flat[2], h_kv_flat[2])
            h_kv_sa_list.append(h_normal_sa)
        if self.use_texture:
            h_texture_sa, _ = self.self_attn_texture(h_kv_flat[3], h_kv_flat[3], h_kv_flat[3])
            h_kv_sa_list.append(h_texture_sa)
        
        # 步骤 3：跨模态 cross-attention
        # K/V: 拼接所有非 CLIP 模态 + modality token
        # 先给每个模态加 modality token
        kv_tokens = []
        for i, h in enumerate(h_kv_sa_list):
            # h: [B, HW, D_u], modality_token: [D_u]
            token = self.modality_tokens[i].unsqueeze(0).unsqueeze(0)  # [1, 1, D_u]
            h_with_token = h + token  # broadcast
            kv_tokens.append(h_with_token)
        kv = torch.cat(kv_tokens, dim=1)  # [B, HW*n_kv, D_u]
        
        h_fused, _ = self.cross_attn(
            query=h_clip_sa,  # [B, HW, D_u]
            key=kv,
            value=kv,
        )
        # h_fused: [B, HW, D_u]
        
        # 步骤 4：模态门控
        gate_logits = self.gate(h_fused)  # [B, HW, n_mods]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 加权求和
        h_mods = [h_clip_sa] + h_kv_sa_list  # [n_mods] 个 [B, HW, D_u]
        h_mods_stack = torch.stack(h_mods, dim=-1)  # [B, HW, D_u, n_mods]
        h_gated = h_fused + (h_mods_stack * gate_weights.unsqueeze(2)).sum(dim=-1)
        
        # 步骤 5：投影压缩
        fused_latent = self.proj_out(h_gated)  # [B, HW, d_latent]
        fused_latent = fused_latent.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # [B, d_latent, H, W]
        
        return fused_latent, gate_weights.reshape(B, H, W, -1)


class FusionDecoder(nn.Module):
    """解码器：从 d_latent 维 fused latent 恢复 512 维 CLIP 嵌入。
    
    用于推理时：3D 渲染出 d_latent 维 latent -> decoder -> 512 维 CLIP -> 文本查询
    """
    def __init__(self, d_latent=16, dim_clip=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, dim_clip),
        )
    
    def forward(self, x):
        # x: [B, d_latent, H, W] -> [B, 512, H, W]
        B, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, d_latent]
        x = self.decoder(x)  # [B, H, W, 512]
        return x.permute(0, 3, 1, 2)  # [B, 512, H, W]
```

### 4.2 baseline 融合方案（用于消融对比）

#### 4.2.1 Concat + MLP（最简方案）

**文件**：`mm_langsplat/fusion/concat_mlp.py`

```python
import torch
import torch.nn as nn

class ConcatMLPFusion(nn.Module):
    """简单拼接 + MLP 投影。baseline 方案。"""
    def __init__(self, dims_dict, d_latent=16):
        super().__init__()
        total_dim = sum(dims_dict.values())
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, d_latent),
        )
    
    def forward(self, feats):
        # 拼接所有模态
        B, _, H, W = feats['clip'].shape
        concat = torch.cat([f.permute(0,2,3,1) for f in feats.values()], dim=-1)
        # [B, H, W, total_dim]
        out = self.mlp(concat)  # [B, H, W, d_latent]
        return out.permute(0, 3, 1, 2), None  # 第二个返回值兼容 cross-attn 接口
```

#### 4.2.2 Transformer Encoder（对称融合）

**文件**：`mm_langsplat/fusion/transformer_encoder.py`

```python
import torch
import torch.nn as nn

class TransformerEncoderFusion(nn.Module):
    """5 个模态 token 过 Transformer Encoder。baseline 方案。"""
    def __init__(self, dims_dict, d_unified=256, d_latent=16, n_heads=8, n_layers=2):
        super().__init__()
        self.projs = nn.ModuleDict({
            k: nn.Linear(v, d_unified) for k, v in dims_dict.items()
        })
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_unified, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_unified, d_latent)
    
    def forward(self, feats):
        B, _, H, W = feats['clip'].shape
        # 每个像素有 5 个 token（5 模态）
        tokens = []
        for k, proj in self.projs.items():
            f = feats[k].permute(0, 2, 3, 1)  # [B, H, W, D_m]
            t = proj(f)  # [B, H, W, d_unified]
            tokens.append(t)
        tokens = torch.stack(tokens, dim=2)  # [B, H, W, 5, d_unified]
        # reshape 给 transformer
        tokens = tokens.reshape(B * H * W, 5, -1)  # [B*H*W, 5, d_u]
        out = self.encoder(tokens)  # [B*H*W, 5, d_u]
        # 平均池化
        out = out.mean(dim=1)  # [B*H*W, d_u]
        out = self.proj_out(out)  # [B*H*W, d_latent]
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out, None
```

#### 4.2.3 MoE（专家混合）

**文件**：`mm_langsplat/fusion/moe.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFusion(nn.Module):
    """Mixture of Experts 融合。每个模态一个 expert。"""
    def __init__(self, dims_dict, d_latent=16, d_unified=256, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.n_experts = len(dims_dict)
        self.experts = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(v, d_unified),
                nn.GELU(),
                nn.Linear(d_unified, d_latent),
            ) for k, v in dims_dict.items()
        })
        # Router: 输入拼接特征，输出每个 expert 的权重
        total_dim = sum(dims_dict.values())
        self.router = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.n_experts),
        )
    
    def forward(self, feats):
        B, _, H, W = feats['clip'].shape
        # 计算每个像素的路由权重
        concat = torch.cat([f.permute(0,2,3,1) for f in feats.values()], dim=-1)
        # [B, H, W, total_dim]
        router_logits = self.router(concat)  # [B, H, W, n_experts]
        weights = F.softmax(router_logits, dim=-1)
        
        # 每个 expert 输出
        out = 0
        for i, (k, expert) in enumerate(self.experts.items()):
            e_out = expert(feats[k].permute(0,2,3,1))  # [B, H, W, d_latent]
            out = out + weights[..., i:i+1] * e_out
        
        return out.permute(0, 3, 1, 2), weights
```

### 4.3 训练融合网络

**文件**：`mm_langsplat/trainers/train_fusion.py`

```python
"""
训练跨模态融合网络。

输入：阶段 3 提取的多模态特征
输出：训练好的融合网络 + 解码器（保存到 pretrained/fusion_net.pt）

损失：
    1. 重构损失：融合 latent 经解码器恢复各模态特征
    2. CLIP 对齐损失：融合 latent 经解码器恢复 CLIP 特征
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mm_langsplat.fusion.cross_attention import CrossModalFusionNetwork, FusionDecoder
from mm_langsplat.data.multimodal_dataset import MultiModalFeatureDataset


def train_fusion(scene_dir, output_path, d_latent=16, epochs=100, lr=5e-4):
    device = torch.device('cuda')
    
    # 数据集
    dataset = MultiModalFeatureDataset(scene_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    # 模型
    fusion_net = CrossModalFusionNetwork(
        dim_clip=512, dim_dino=1024, dim_depth=1,
        dim_normal=3, dim_texture=64,
        d_unified=256, d_latent=d_latent,
    ).to(device)
    
    decoder = FusionDecoder(d_latent=d_latent, dim_clip=512).to(device)
    
    # 优化器
    params = list(fusion_net.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    
    # 损失函数
    mse_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss()
    
    # 训练循环
    for epoch in range(epochs):
        for batch in dataloader:
            feats = {k: v.to(device) for k, v in batch.items()}
            
            # 前向
            fused_latent, gate_weights = fusion_net(feats)
            # fused_latent: [B, d_latent, H, W]
            
            # 解码
            recon_clip = decoder(fused_latent)  # [B, 512, H, W]
            
            # 损失 1：CLIP 对齐（核心）
            B, _, H, W = recon_clip.shape
            loss_align = cos_loss(
                recon_clip.permute(0,2,3,1).reshape(-1, 512),
                feats['clip'].permute(0,2,3,1).reshape(-1, 512),
                torch.ones(B*H*W).to(device),
            )
            
            # 损失 2：重构（可选，作为正则）
            # ... 可加 DINO/depth/normal 的重构损失
            
            loss = loss_align  # + 0.1 * loss_recon
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")
    
    # 保存
    torch.save({
        'fusion_net': fusion_net.state_dict(),
        'decoder': decoder.state_dict(),
        'd_latent': d_latent,
    }, output_path)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='pretrained/fusion_net.pt')
    parser.add_argument('--d_latent', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    train_fusion(args.scene_dir, args.output, args.d_latent, args.epochs)
```

### 4.4 验收

- 训练损失收敛（loss < 0.1）
- 融合网络参数量 < 10M
- 单次 forward 时间 < 100 ms（在 RTX 3090 上）
- 模型大小 < 50 MB

---

## 第五阶段：多模态语言高斯模型实现

### 5.1 修改 LangSplat 的 GaussianModel

**原文件**：`scene/gaussian_model.py`
**新文件**：`mm_langsplat/models/mm_language_gaussian.py`

```python
"""
多模态语言高斯模型。
继承自 LangSplat 的 GaussianModel，扩展：
    - 每个 3D 高斯额外存 d_latent 维语言 latent（三层级 s/p/w 各一套）
    - 渲染时用 tile-based rasterizer（复用 LangSplat）
    - 监督信号是阶段 4 训练好的融合 latent
"""

import torch
import torch.nn as nn
import numpy as np
from scene.gaussian_model import GaussianModel


class MMLanguageGaussianModel(GaussianModel):
    """扩展 GaussianModel，加入多模态语言 latent。"""
    
    def __init__(self, sh_degree=3, d_latent=16, n_levels=3):
        super().__init__(sh_degree)
        self.d_latent = d_latent
        self.n_levels = n_levels  # SAM 三层级
        
        # 三层级语言 latent
        # 每个高斯存 d_latent * 3 维
        self.language_latents = nn.ParameterList([
            nn.Parameter(torch.zeros(1, d_latent))  # 初始化为 1 个点，后续动态扩展
            for _ in range(n_levels)
        ])
    
    def create_from_pcd(self, pcd):
        """从点云初始化高斯，同时初始化语言 latent。"""
        super().create_from_pcd(pcd)
        n_points = pcd.shape[0]
        # 初始化三层级 latent
        for i in range(self.n_levels):
            self.language_latents[i] = nn.Parameter(
                torch.zeros(n_points, self.d_latent, device='cuda') * 0.01
            )
    
    def train_language(self, fusion_net, decoder, train_views, 
                       iterations=30000, lr=5e-4):
        """训练语言 latent（冻结其他高斯参数）。
        
        Args:
            fusion_net: 阶段 4 训练好的融合网络
            decoder: 阶段 4 训练好的解码器
            train_views: 训练视角列表
        """
        # 冻结其他参数
        for name, param in self.named_parameters():
            if 'language_latents' not in name:
                param.requires_grad = False
        
        # 只训 language latent
        params = [p for p in self.language_latents]
        optimizer = torch.optim.AdamW(params, lr=lr)
        
        for iter in range(iterations):
            view = train_views[iter % len(train_views)]
            
            # 加载该视角的多模态特征
            feats = load_multimodal_features(view)
            
            # 用融合网络生成监督信号
            with torch.no_grad():
                target_latent, _ = fusion_net(feats)  # [B, d_latent, H, W]
            
            # 渲染三层级 latent
            rendered_latents = []
            for level in range(self.n_levels):
                # 修改 rasterizer，把高斯的 language_latents 渲染到 2D
                # 复用 LangSplat 的 rasterizer，把 f 替换为 language_latents[level]
                rendered = self.rasterize_language(view, level)  # [B, d_latent, H, W]
                rendered_latents.append(rendered)
            
            # 损失：渲染 latent 与监督 latent 一致
            loss = 0
            for level in range(self.n_levels):
                loss += F.mse_loss(rendered_latents[level], target_latent)
                # 加 cosine loss
                loss += 0.1 * (1 - F.cosine_similarity(
                    rendered_latents[level], target_latent, dim=1
                ).mean())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter % 1000 == 0:
                print(f"Iter {iter}, loss={loss.item():.4f}")
    
    def rasterize_language(self, view, level):
        """渲染指定层级的语言 latent。
        
        复用 LangSplat 的 tile-based rasterizer，输入高斯的语言 latent 而非颜色。
        """
        # 调用修改版 diff-gaussian-rasterization
        # 见第六阶段 6.3 节关于修改 rasterizer 的说明
        pass
```

### 5.2 修改 train.py

**修改文件**：`train.py`（LangSplat 原版）
**新增文件**：`mm_langsplat/trainers/train_langsplat_mm.py`

```python
"""
多模态语言高斯训练入口。
"""

import os
import torch
from scene import Scene
from mm_langsplat.models.mm_language_gaussian import MMLanguageGaussianModel
from mm_langsplat.fusion.cross_attention import CrossModalFusionNetwork, FusionDecoder
from argparse import ArgumentParser


def train_mm_langsplat(args):
    # 1. 加载已训练的 RGB 场景
    gaussians = MMLanguageGaussianModel(
        sh_degree=3,
        d_latent=args.d_latent,
        n_levels=3
    )
    gaussians.load_ply(args.rgb_model_path)  # LangSplat 训好的 RGB 模型
    
    # 2. 加载训练好的融合网络
    ckpt = torch.load(args.fusion_net_path)
    fusion_net = CrossModalFusionNetwork(d_latent=args.d_latent)
    fusion_net.load_state_dict(ckpt['fusion_net'])
    fusion_net.eval()
    
    decoder = FusionDecoder(d_latent=args.d_latent)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.eval()
    
    # 3. 训练语言 latent
    scene = Scene(args, load_iteration=-1)  # 加载 LangSplat 场景
    train_views = scene.getTrainCameras()
    
    gaussians.train_language(
        fusion_net=fusion_net,
        decoder=decoder,
        train_views=train_views,
        iterations=args.iterations,
        lr=args.lr,
    )
    
    # 4. 保存
    gaussians.save_language(os.path.join(args.model_path, 'mm_langsplat.ply'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rgb_model_path', type=str, required=True)
    parser.add_argument('--fusion_net_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--d_latent', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    train_mm_langsplat(args)
```

---

## 第六阶段：光栅化器（Rasterizer）适配

### 6.1 修改 diff-gaussian-rasterization

LangSplat 用的 `submodules/diff-gaussian-rasterization` 默认只渲染 RGB（3 维）。需要扩展为渲染任意维度特征。

**方式 1（推荐）**：用 LangSplat 官方 fork 版本，已支持任意维度。

```bash
# LangSplat 仓库已自带支持任意维度的 diff-gaussian-rasterization
# 路径：submodules/diff-gaussian-rasterization-feature
# 见 LangSplat README 中的安装说明
```

### 6.2 修改 rasterizer 调用

**修改文件**：`gaussian_renderer/__init__.py`

```python
# 原版渲染 RGB：
# rendered_image = rasterizer(...)

# 修改为：渲染 d_latent 维语言特征
def render_language(viewpoint_camera, pc, level, pipe, bg, d_latent):
    """渲染指定层级的语言 latent。
    
    Args:
        level: 0=subpart, 1=part, 2=whole
        d_latent: 语言 latent 维度
    """
    # 获取该层级的语言 latent 作为 "color"
    language_latent = pc.language_latents[level]  # [N, d_latent]
    
    # 调用 rasterizer
    raster_settings = GaussianRasterizationSettings(
        image_height=viewpoint_camera.image_height,
        image_width=viewpoint_camera.image_width,
        tanfovxhalf=viewpoint_camera.FoVx,
        tanfovyhalf=viewpoint_camera.FoVy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,  # 语言 latent 不用 SH
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # 渲染
    rendered_latent, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=pc.get_xy,
        opacities=pc.get_opacity,
        shs=None,  # 不用 SH
        colors_precomp=language_latent,  # 用 latent 替代 color
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    # rendered_latent: [d_latent, H, W]
    return rendered_latent
```

### 6.3 验收

- rasterizer 能正确渲染 $d_{latent}$ 维特征
- 渲染速度与原版 RGB 渲染相当（< 50 ms/帧 @1080p）
- 梯度能正确反传到高斯 latent

---

## 第七阶段：推理与评测

### 7.1 推理流程

**文件**：`mm_langsplat/inference.py`

```python
"""
推理流程：
    1. 加载训练好的 MM-LangSplat 模型
    2. 加载融合网络 + 解码器
    3. 对查询文本：CLIP text encoder -> phi_qry
    4. 渲染 3D 高斯 latent (3 层级) -> F_l
    5. 解码器：F_l -> 512 维 CLIP 嵌入
    6. 计算关联度评分 -> 选最优层级
    7. 输出定位/分割结果
"""

import torch
import open_clip
from mm_langsplat.models.mm_language_gaussian import MMLanguageGaussianModel
from mm_langsplat.fusion.cross_attention import FusionDecoder


def query_3d(model_path, fusion_net_path, text_query, scene_data):
    """开放词汇 3D 查询。"""
    device = torch.device('cuda')
    
    # 1. 加载模型
    gaussians = MMLanguageGaussianModel(d_latent=16)
    gaussians.load_ply(model_path)
    
    ckpt = torch.load(fusion_net_path)
    decoder = FusionDecoder(d_latent=16)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.eval()
    
    # 2. CLIP 文本编码
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    text_token = tokenizer(text_query).to(device)
    with torch.no_grad():
        phi_qry = clip_model.encode_text(text_token)  # [1, 512]
    
    # 3. 渲染三层级 latent
    relevancy_maps = []
    for level in range(3):  # subpart, part, whole
        rendered_latent = gaussians.rasterize_language(scene_data['camera'], level)
        # rendered_latent: [d_latent, H, W]
        
        # 4. 解码到 512 维 CLIP
        with torch.no_grad():
            rendered_clip = decoder(rendered_latent.unsqueeze(0))
        # rendered_clip: [1, 512, H, W]
        
        # 5. 关联度评分（沿用 LangSplat 公式 7）
        relevancy = compute_relevancy(rendered_clip, phi_qry)
        relevancy_maps.append(relevancy)
    
    # 6. 选最优层级
    best_level = max(range(3), key=lambda i: relevancy_maps[i].max())
    
    return relevancy_maps[best_level], best_level


def compute_relevancy(rendered_clip, phi_qry, canon_phrases=['object', 'things', 'stuff', 'texture']):
    """计算关联度评分，沿用 LangSplat 公式 (7)。"""
    # ... 复用 LangSplat 的 relevancy 计算逻辑
    pass
```

### 7.2 评测脚本

**文件**：`mm_langsplat/eval_mm_langsplat.py`

```python
"""
评测脚本。
指标：
    - LERF: Localization Accuracy (%), mIoU (%)
    - 3D-OVS: mIoU (%), Accuracy (%)
对比 SOTA：LangSplat, Occam's LGS, ILGS
"""

import os
import json
import torch
from mm_langsplat.inference import query_3d


def evaluate_lerf(model_path, fusion_net_path, lerf_dir):
    """评测 LERF 数据集。"""
    results = {}
    
    for scene in ['ramen', 'figurines', 'teatime', 'waldo_kitchen']:
        scene_dir = os.path.join(lerf_dir, scene)
        
        # 加载评测查询
        queries = load_lerf_queries(scene_dir)
        
        loc_correct = 0
        seg_ious = []
        
        for q in queries:
            text = q['text']
            gt_loc = q['location']
            gt_mask = q['mask']
            
            # 推理
            relevancy_map, level = query_3d(
                model_path, fusion_net_path, text, scene_dir
            )
            
            # 定位准确率
            pred_loc = relevancy_map.argmax()
            if distance(pred_loc, gt_loc) < threshold:
                loc_correct += 1
            
            # 分割 IoU
            pred_mask = (relevancy_map > 0.5).float()
            iou = (pred_mask * gt_mask).sum() / (pred_mask + gt_mask - pred_mask * gt_mask).sum()
            seg_ious.append(iou)
        
        results[scene] = {
            'loc_accuracy': loc_correct / len(queries),
            'mIoU': sum(seg_ious) / len(seg_ious),
        }
    
    # 汇总
    overall_loc = sum(r['loc_accuracy'] for r in results.values()) / 4
    overall_miou = sum(r['mIoU'] for r in results.values()) / 4
    results['overall'] = {
        'loc_accuracy': overall_loc,
        'mIoU': overall_miou,
    }
    
    return results


if __name__ == '__main__':
    results = evaluate_lerf(
        model_path='pretrained/mm_langsplat_lerf_ramen.ply',
        fusion_net_path='pretrained/fusion_net.pt',
        lerf_dir='data/LERF',
    )
    print(json.dumps(results, indent=2))
```

### 7.3 评测指标

| 数据集 | 任务 | 指标 | LangSplat baseline | MM-LangSplat 目标 |
|--------|------|------|-------------------|-------------------|
| LERF | 3D 物体定位 | Accuracy (%) | 84.3 | **>88** |
| LERF | 3D 语义分割 | mIoU (%) | 51.4 | **>60** |
| 3D-OVS | 3D 语义分割 | mIoU (%) | 93.4 | **>95** |
| 3D-OVS | 3D 语义分割 | Accuracy (%) | 98.9 | **>99** |

---

## 第八阶段：消融实验与维度搜索

### 8.1 模态消融（实验 1）

逐个去掉模态，测 LERF mIoU：

| 配置 | CLIP | DINOv2 | Depth | Normal | Texture | 预期 mIoU |
|------|------|--------|-------|--------|---------|-----------|
| baseline | ✓ | ✗ | ✗ | ✗ | ✗ | 51.4 |
| + DINOv2 | ✓ | ✓ | ✗ | ✗ | ✗ | ~55 |
| + Depth | ✓ | ✗ | ✓ | ✗ | ✗ | ~54 |
| + DINOv2 + Depth | ✓ | ✓ | ✓ | ✗ | ✗ | ~58 |
| + Normal | ✓ | ✓ | ✓ | ✓ | ✗ | ~60 |
| + Texture (full) | ✓ | ✓ | ✓ | ✓ | ✓ | ~65 |

**命令**：
```bash
# 各配置跑一遍
for config in "use_clip only" "use_clip+dino" "use_clip+depth" "use_clip+dino+depth" "use_clip+dino+depth+normal" "all"; do
    python -m mm_langsplat.trainers.train_langsplat_mm \
        --config configs/$config.yml \
        --scene_dir data/LERF/ramen
    python -m mm_langsplat.eval_mm_langsplat --config configs/$config.yml
done
```

### 8.2 融合策略消融（实验 2）

对比 4 种融合方法：

| 融合方法 | LERF mIoU（预期） | 推理速度 | 参数量 |
|---------|------------------|---------|--------|
| Concat + MLP | ~58 | 最快 | 最少 |
| Transformer Encoder | ~62 | 中 | 中 |
| **Cross-Attention（主推）** | **~65** | 中 | 中 |
| MoE | ~63 | 中 | 较多 |

### 8.3 维度 $d$ 搜索（实验 3）

网格搜索 $d \in \{4, 8, 16, 32, 64\}$：

| $d$ | LERF mIoU（预期） | 显存 | 速度 |
|-----|------------------|------|------|
| 4 | ~60 | 30 MB | 最快 |
| 8 | ~63 | 60 MB | 快 |
| **16（推荐）** | **~65** | **120 MB** | **中** |
| 32 | ~66 | 240 MB | 中-慢 |
| 64 | ~65（过拟合） | 480 MB | 慢 |

```bash
# 维度搜索
for d in 4 8 16 32 64; do
    python -m mm_langsplat.trainers.train_fusion \
        --scene_dir data/LERF/ramen \
        --output pretrained/fusion_d${d}.pt \
        --d_latent $d
    python -m mm_langsplat.trainers.train_langsplat_mm \
        --fusion_net_path pretrained/fusion_d${d}.pt \
        --d_latent $d \
        --model_path output/mm_langsplat_d${d}
    python -m mm_langsplat.eval_mm_langsplat \
        --model_path output/mm_langsplat_d${d}/mm_langsplat.ply \
        --d_latent $d
done
```

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

## 第十阶段：核心论文与引用清单（汇总）

### 10.1 基础框架论文

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| LangSplat: 3D Language Gaussian Splatting (CVPR 2024) | Qin et al. | https://arxiv.org/abs/2312.16084 | https://github.com/minghanqin/LangSplat |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023) | Kerbl et al. | https://repo.sam.inria.fr/fungraph/3d-gaussian-splatting/ | https://github.com/graphdeco-inria/gaussian-splatting |
| Segment Anything (ICCV 2023) | Kirillov et al. | https://arxiv.org/abs/2304.02643 | https://github.com/facebookresearch/segment-anything |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP, ICML 2021) | Radford et al. | https://arxiv.org/abs/2103.00020 | https://github.com/openai/CLIP |
| LERF: Language Embedded Radiance Fields (ICCV 2023) | Kerr et al. | https://lerf.io | https://github.com/lerftoy/lerf |

### 10.2 多模态特征提取论文

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| DINOv2: Learning Robust Visual Features without Supervision (TMLR 2024) | Oquab et al. | https://arxiv.org/abs/2304.07193 | https://github.com/facebookresearch/dinov2 |
| Depth Anything V2 (NeurIPS 2024) | Yang et al. | https://arxiv.org/abs/2406.09414 | https://github.com/DepthAnything/Depth-Anything-V2 |
| DSINE: Revisiting Depth-aware Normal Estimation (ECCV 2024) | Bae et al. | https://arxiv.org/abs/2403.18205 | https://github.com/baegwangbin/DSINE |
| Unsupervised Semantic Segmentation by Distilling Feature Correspondences (STEGO, ICLR 2022) | Hamilton et al. | https://arxiv.org/abs/2207.05026 | https://github.com/hamarb172/STEGO |

### 10.3 多模态融合参考论文（创新点对标）

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields (CVPR 2024) | Zhou et al. | https://arxiv.org/abs/2312.03203 | https://github.com/ShijieZhou-UCLA/feature-3dgs |
| 3D Vision-Language Gaussian Splatting (ICLR 2025) | Peng et al. | https://arxiv.org/abs/2410.07577 | https://github.com/3d-vlgs |
| SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields (arXiv 2025) | Li et al. | https://arxiv.org/abs/2506.09565 | https://semanticsplat.github.io |
| ShelfGaussian: Shelf-Supervised Open-Vocabulary Gaussian-Based 3D Scene Understanding (CVPR 2026) | Zhao et al. | https://cvpr.thecvf.com/virtual/2026/events/Highlights2026 | https://lunarlab-gatech.github.io |
| FMGS: Foundation Model Embedded 3D Gaussian Splatting (ICLR 2024) | Zuo et al. | https://arxiv.org/abs/2401.01970 | https://github.com/xingxingzuo/FMGS |

### 10.4 融合架构参考论文

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| Attention Is All You Need (NeurIPS 2017) | Vaswani et al. | https://arxiv.org/abs/1706.03762 | — |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs (ICLR 2022) | Jaegle et al. | https://arxiv.org/abs/2107.14795 | https://github.com/deepmind/deepmind-research/tree/master/perceiver |
| Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (ICLR 2017) | Shazeer et al. | https://arxiv.org/abs/1701.06538 | — |

### 10.5 数据集论文

| 数据集 | 引用 | 链接 | GitHub |
|--------|------|------|--------|
| LERF Dataset (ICCV 2023) | Kerr et al. | https://lerf.io | https://github.com/lerftoy/lerf |
| 3D-OVS Dataset (NeurIPS 2023) | Liu et al. | https://arxiv.org/abs/2312.03203 | https://github.com/Kunhao-Liu/3D-OVS |

---

## 第十一阶段：调试与故障排除

### 11.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OOM during DINOv2 inference | ViT-L 模型大 | 改用 `dinov2_vitb14`（768d）或 `dinov2_vits14`（384d） |
| DSINE 输出法向量全 0 | 输入未归一化 | 用 ImageNet mean/std 归一化：`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` |
| Cross-attention loss 不收敛 | 学习率太高 | 降到 `1e-5`，加 warmup 1000 步 |
| 渲染 latent 全 0 | rasterizer 不支持非 3 通道 | 用 LangSplat 自带的 `diff-gaussian-rasterization-feature` |
| mIoU 不升 | 融合网络未训好 | 增加 epochs 到 200；检查 CLIP 对齐损失 |
| 训练显存超 | 5 模态全开 + 2.5M 高斯 | 改用 fp16；减小 `d_latent` 到 8；分阶段加载 |

### 11.2 调试技巧

```python
# 检查各模态特征是否合理
import matplotlib.pyplot as plt

def visualize_features(feats):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (k, v) in zip(axes, feats.items()):
        # PCA 降到 3 维可视化
        v_3d = pca(v[0].permute(1,2,0).reshape(-1, v.shape[1]).cpu().numpy(), 3)
        v_3d = v_3d.reshape(v.shape[2], v.shape[3], 3)
        ax.imshow(v_3d)
        ax.set_title(k)
    plt.savefig('feature_visualization.png')

# 检查 gate weights
def visualize_gate(gate_weights):
    # gate_weights: [B, H, W, n_mods]
    fig, axes = plt.subplots(1, n_mods, figsize=(5*n_mods, 5))
    for i, ax in enumerate(axes):
        ax.imshow(gate_weights[0,:,:,i].cpu().numpy(), cmap='hot')
        ax.set_title(f'Modality {i}')
    plt.savefig('gate_weights.png')
```

---

## 第十二阶段：验收与交付

### 12.1 验收清单

- [ ] 阶段 0：LangSplat baseline 复现，LERF mIoU ≈ 51.4%
- [ ] 阶段 1：所有 5 模态提取器可独立加载并输出正确维度
- [ ] 阶段 2：`MultiModalExtractor` 联合提取可运行
- [ ] 阶段 3：单场景多模态预处理完成，文件结构正确
- [ ] 阶段 4：融合网络训练 loss 收敛（< 0.1）
- [ ] 阶段 5-6：语言高斯训练完成，3D 渲染可生成 d_latent 维 latent
- [ ] 阶段 7：推理可生成 relevancy map
- [ ] 阶段 7：评测 LERF mIoU > 60%（超越 LangSplat baseline 51.4%）
- [ ] 阶段 8：消融实验完成（模态消融 + 融合策略对比 + 维度搜索）
- [ ] 阶段 8：3D-OVS mIoU > 95%（超越 LangSplat baseline 93.4%）

### 12.2 最终交付物

```
output/
├── code/
│   ├── mm_langsplat/                    # 多模态融合模块
│   ├── modified_train.py               # 修改版训练入口
│   └── modified_preprocess.py          # 修改版预处理
├── pretrained/
│   ├── fusion_lerf_ramen.pt
│   ├── fusion_lerf_figurines.pt
│   ├── ... (其他场景)
│   └── mm_langsplat_lerf_ramen.ply
├── results/
│   ├── lerf_results.json               # LERF 评测结果
│   ├── 3dovs_results.json              # 3D-OVS 评测结果
│   ├── ablation_modality.csv           # 模态消融
│   ├── ablation_fusion.csv             # 融合策略对比
│   └── ablation_dim.csv                # 维度搜索
└── figures/
    ├── pareto_curve.png                # mIoU vs 显存 帕累托曲线
    ├── gate_weights_visualization.png   # 模态权重可视化
    └── qualitative_comparison.png      # 与 SOTA 对比可视化
```

---

## 附录 A：核心改动文件清单

| 原文件 | 新文件 | 改动类型 |
|--------|--------|---------|
| `preprocess.py` | `mm_langsplat/preprocess_mm.py` | 新增：5 模态特征提取 |
| `train.py` | `mm_langsplat/trainers/train_langsplat_mm.py` | 新增：训练多模态语言高斯 |
| `gaussian_renderer/__init__.py` | `mm_langsplat/render_language.py` | 新增：渲染语言 latent |
| `scene/gaussian_model.py` | `mm_langsplat/models/mm_language_gaussian.py` | 继承：加 language_latents |
| `eval_langsplat.py` | `mm_langsplat/eval_mm_langsplat.py` | 新增：评测多模态版本 |
| — | `mm_langsplat/extractors/*.py` | 新增：5 个模态提取器 |
| — | `mm_langsplat/fusion/*.py` | 新增：4 种融合策略 |
| — | `mm_langsplat/data/multimodal_dataset.py` | 新增：数据加载器 |
| — | `scripts/run_full_pipeline.sh` | 新增：一键执行脚本 |

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
> 4. 最终目标：LERF mIoU > 60%，3D-OVS mIoU > 95%

---
*AI生成*
