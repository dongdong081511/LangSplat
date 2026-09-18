"""Tile-level Cross-Modal Fusion Network.

Takes CLIP + DINOv2 per-tile features and outputs fused 512-dim features
(CLIP-compatible). Uses cross-attention at tile level (~300 tiles, O(300^2) safe)
to avoid the pixel-level OOM problem (480x640=307200 tokens, ~376GB attention).

Architecture:
    1. Project CLIP (512) and DINOv2 (768) to unified dim (256)
    2. Self-attention on CLIP tiles (cross-tile context)
    3. Cross-attention: CLIP as query, DINOv2 as K/V
    4. Gated fusion + output projection to 512

Training:
    Input: CLIP (with dropout), DINOv2
    Target: CLIP (original, no dropout)
    Dropout forces network to learn from DINOv2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TileCrossAttentionFusion(nn.Module):
    """Tile-level cross-attention fusion for CLIP + DINOv2.

    Args:
        dim_clip: CLIP feature dim (512)
        dim_dino: DINOv2 feature dim (768 for vitb14, 1024 for vitl14)
        d_unified: unified intermediate dim (256)
        d_out: output dim (512, CLIP-compatible)
        n_heads: attention heads (4)
        dropout: CLIP input dropout rate during training (0.3)
    """

    def __init__(self, dim_clip=512, dim_dino=768, d_unified=256,
                 d_out=512, n_heads=4, dropout=0.3):
        super().__init__()
        self.dim_clip = dim_clip
        self.dim_dino = dim_dino
        self.d_unified = d_unified
        self.d_out = d_out

        # Project modalities to unified dim
        self.proj_clip = nn.Sequential(
            nn.Linear(dim_clip, d_unified),
            nn.LayerNorm(d_unified),
        )
        self.proj_dino = nn.Sequential(
            nn.Linear(dim_dino, d_unified),
            nn.LayerNorm(d_unified),
        )

        # Self-attention on CLIP tiles (cross-tile context)
        self.self_attn = nn.MultiheadAttention(
            d_unified, n_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_unified)

        # Cross-attention: CLIP query, DINOv2 as K/V
        self.cross_attn = nn.MultiheadAttention(
            d_unified, n_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(d_unified)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_unified, d_unified * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_unified * 2, d_unified),
        )
        self.norm3 = nn.LayerNorm(d_unified)

        # Output projection to d_out
        self.proj_out = nn.Sequential(
            nn.Linear(d_unified, d_unified),
            nn.GELU(),
            nn.Linear(d_unified, d_out),
        )

        # CLIP input dropout for training (forces learning from DINOv2)
        self.clip_dropout = nn.Dropout(dropout)

    def forward(self, clip_feat, dino_feat, is_training=False):
        """Fuse CLIP and DINOv2 features.

        Args:
            clip_feat: [num_tiles, 512] CLIP features (unit normalized)
            dino_feat: [num_tiles, 768] DINOv2 features (unit normalized)
            is_training: if True, apply dropout to CLIP input

        Returns:
            fused: [num_tiles, 512] fused features (unit normalized)
        """
        # Dropout on CLIP during training
        if is_training:
            clip_input = self.clip_dropout(clip_feat)
        else:
            clip_input = clip_feat

        # Project to unified dim
        q = self.proj_clip(clip_input)   # [num_tiles, d_unified]
        kv = self.proj_dino(dino_feat)   # [num_tiles, d_unified]

        # Add batch dimension for MultiheadAttention (expects [B, N, D])
        q_b = q.unsqueeze(0)   # [1, num_tiles, d_unified]
        kv_b = kv.unsqueeze(0)  # [1, num_tiles, d_unified]

        # Self-attention on CLIP (cross-tile context)
        attn_out, _ = self.self_attn(q_b, q_b, q_b)
        q = self.norm1(q_b + attn_out)  # [1, num_tiles, d_unified]

        # Cross-attention: CLIP queries, DINOv2 as K/V
        cross_out, _ = self.cross_attn(q, kv_b, kv_b)
        q = self.norm2(q + cross_out)  # [1, num_tiles, d_unified]

        # FFN
        q = self.norm3(q + self.ffn(q))  # [1, num_tiles, d_unified]

        # Output projection
        out = self.proj_out(q.squeeze(0))  # [num_tiles, d_out]

        # Normalize
        out = F.normalize(out, dim=-1)
        return out


def fusion_loss(pred, target):
    """Combined MSE + cosine similarity loss.

    Args:
        pred: [N, 512] predicted features
        target: [N, 512] target CLIP features
    """
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    return mse + cos


def mse_only_loss(pred, target):
    """MSE-only loss (cos_sim removed). For unit-normalized outputs this is
    approximately equivalent to 2*(1-cos_sim), so it mainly re-weights the
    reconstruction objective. Kept for ablation comparison."""
    return F.mse_loss(pred, target)


def info_nce_loss(pred, target, temperature=0.07):
    """Cross-modal InfoNCE contrastive loss.

    Treats each tile's fused feature as the anchor, its matching CLIP feature
    as the positive, and all other tiles' CLIP features in the same image as
    negatives. This forces the fused features to preserve tile-level
    discriminability (not just copy CLIP) while staying aligned with CLIP space.

    Args:
        pred: [N, 512] fused (anchor) features, unit-normalized
        target: [N, 512] CLIP target features, unit-normalized
        temperature: softmax temperature (CLIP default 0.07)
    """
    # Cosine similarity matrix [N, N]: pred_i vs target_j
    sim = pred @ target.t() / temperature
    # Positive: diagonal entries (pred_i vs target_i)
    # Loss = -log( exp(sim_ii/T) / sum_j exp(sim_ij/T) )
    labels = torch.arange(pred.size(0), device=pred.device)
    loss = F.cross_entropy(sim, labels)
    return loss
