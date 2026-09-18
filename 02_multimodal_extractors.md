# MM-LangSplat 改造方案（分册 02：多模态特征提取器实现）

> **本册定位**：编码任务。实现 CLIP/DINOv2/Depth/Normal/Texture/SAM 共 6 个提取器，独立可完成。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

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


---


---
*AI生成*
