# MM-LangSplat 改造方案（分册 04：Cross-Attention 融合网络）

> **本册定位**：核心创新。实现 4 种融合策略（主推 HCF + 3 个 baseline），训练融合网络。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

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


---


---
*AI生成*
