# MM-LangSplat 改造方案（分册 05：3D 语言高斯与光栅化器改造）

> **本册定位**：强耦合任务。扩展 GaussianModel 支持 language_latents，改 rasterizer 支持任意维度渲染。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

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


---


---
*AI生成*
