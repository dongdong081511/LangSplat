"""Train the tile-level cross-attention fusion network.

Loads CLIP (512-dim) and DINOv2 (768-dim) features per tile from
language_features_mm/ (or any directory with _f.npy and _f_dino.npy files).

Trains TileCrossAttentionFusion: input (CLIP with dropout, DINOv2) -> target CLIP.
The dropout forces the network to learn from DINOv2, not just copy CLIP.

Usage:
    python -m mm_langsplat.train_fusion \
        --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
        --save_path mm_langsplat/ckpt/fusion_teatime.pth \
        --epochs 200 --lr 1e-4 --batch_tiles 256
"""
import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm_langsplat.fusion.cross_attention import (
    TileCrossAttentionFusion, fusion_loss, mse_only_loss, info_nce_loss,
)

# Loss registry
LOSS_FNS = {
    'mse_cos': fusion_loss,   # original: MSE + (1-cos_sim)
    'mse_only': mse_only_loss,
    'infonce': info_nce_loss,
}


def load_features_per_image(feat_dir, use_depth=False):
    """Load CLIP and DINOv2 features per image.

    Returns:
        list of (clip_feat, dino_feat, depth_feat) tuples, one per image.
        clip_feat: [num_tiles, 512] (valid tiles only)
        dino_feat: [num_tiles, 768]
        depth_feat: [num_tiles, 8] or None
    """
    f_files = sorted(glob.glob(os.path.join(feat_dir, '*_f.npy')))
    # Exclude _f_dino.npy / _f_depth.npy
    f_files = [f for f in f_files if f.endswith('_f.npy')]

    data = []
    for f_path in tqdm(f_files, desc='Loading features'):
        dino_path = f_path.replace('_f.npy', '_f_dino.npy')
        if not os.path.exists(dino_path):
            print(f'WARNING: DINO file not found: {dino_path}, skipping')
            continue

        depth_path = f_path.replace('_f.npy', '_f_depth.npy')
        if use_depth and not os.path.exists(depth_path):
            print(f'WARNING: depth file not found: {depth_path}, skipping')
            continue

        clip_feat = np.load(f_path)  # [max_tiles, 512]
        dino_feat = np.load(dino_path)  # [max_tiles, 768]
        depth_feat = np.load(depth_path) if use_depth else None  # [max_tiles, 8]

        # Filter out zero-padded tiles
        clip_norms = np.linalg.norm(clip_feat, axis=1)
        valid = clip_norms > 0.1
        clip_valid = clip_feat[valid]
        dino_valid = dino_feat[valid]
        depth_valid = depth_feat[valid] if use_depth else None

        if len(clip_valid) > 0:
            data.append((clip_valid, dino_valid, depth_valid))

    total_tiles = sum(len(c) for c, _, _ in data)
    print(f'Loaded {total_tiles} tiles from {len(data)} images')
    return data


def train_fusion(feat_dir, save_path, epochs=200, lr=1e-4,
                 dropout=0.3, d_unified=256, n_heads=4, loss_type='mse_cos',
                 temperature=0.07, use_depth=False, dim_depth=8,
                 depth_loss_weight=1.0):
    """Train the fusion network.

    Processes per-image: cross-attention is within each image's tiles.
    This is critical because attention across different images' tiles is meaningless.

    Args:
        feat_dir: directory with _f.npy, _f_dino.npy (and _f_depth.npy if use_depth)
        save_path: path to save fusion network weights
        epochs: training epochs
        lr: learning rate
        dropout: CLIP input dropout rate
        d_unified: unified intermediate dimension
        n_heads: attention heads
        loss_type: one of 'mse_cos' (original), 'mse_only', 'infonce'
        temperature: InfoNCE softmax temperature (only used if loss_type='infonce')
        use_depth: enable third modality (depth statistics)
        dim_depth: depth statistic dimension (8)
        depth_loss_weight: weight of the auxiliary depth reconstruction loss
    """
    import torch.nn.functional as F
    device = torch.device('cuda')

    loss_fn = LOSS_FNS[loss_type]
    print(f'Using loss: {loss_type} | depth modality: {use_depth} '
          f'(aux weight: {depth_loss_weight})')

    # Load per-image features
    data = load_features_per_image(feat_dir, use_depth=use_depth)
    num_images = len(data)
    total_tiles = sum(len(c) for c, _, _ in data)
    print(f'Total: {num_images} images, {total_tiles} tiles')

    # Move to GPU
    data_gpu = []
    for clip_feat, dino_feat, depth_feat in data:
        data_gpu.append((
            torch.from_numpy(clip_feat).float().to(device),
            torch.from_numpy(dino_feat).float().to(device),
            torch.from_numpy(depth_feat).float().to(device) if use_depth else None,
        ))

    # Create fusion network
    model = TileCrossAttentionFusion(
        dim_clip=512, dim_dino=768,
        dim_depth=dim_depth if use_depth else 0,
        d_unified=d_unified, d_out=512,
        n_heads=n_heads, dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop: per-image processing
    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        perm = torch.randperm(num_images)
        total_loss = 0.0
        total_cos_sim = 0.0
        num_processed = 0

        for img_idx in perm:
            clip_feat, dino_feat, depth_feat = data_gpu[img_idx]

            # Forward (training mode with dropout)
            pred, decoded_depth = model(clip_feat, dino_feat, depth_feat,
                                        is_training=True, return_depth=True)
            if loss_type == 'infonce':
                loss = loss_fn(pred, clip_feat, temperature=temperature)
            else:
                loss = loss_fn(pred, clip_feat)  # target = original CLIP (no dropout)
            if depth_feat is not None and decoded_depth is not None:
                depth_loss = F.mse_loss(decoded_depth, depth_feat)
                loss = loss + depth_loss_weight * depth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred, clip_feat, dim=-1).mean()
                total_cos_sim += cos_sim.item()
            num_processed += 1

        scheduler.step()
        avg_loss = total_loss / num_processed
        avg_cos = total_cos_sim / num_processed

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | '
                  f'Train cos_sim: {avg_cos:.4f}')

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'dim_clip': 512, 'dim_dino': 768,
            'dim_depth': dim_depth if use_depth else 0,
            'd_unified': d_unified, 'd_out': 512,
            'n_heads': n_heads, 'dropout': dropout
        }
    }, save_path)
    print(f'Fusion model saved to {save_path}')

    # Final evaluation
    model.eval()
    all_cos = []
    with torch.no_grad():
        for clip_feat, dino_feat, depth_feat in data_gpu:
            pred = model(clip_feat, dino_feat, depth_feat, is_training=False)
            cos_sim = F.cosine_similarity(pred, clip_feat, dim=-1).mean()
            all_cos.append(cos_sim.item())
    print(f'Final: avg cos_sim(fused, CLIP) = {np.mean(all_cos):.4f} +/- {np.std(all_cos):.4f}')
    print(f'  (cos_sim close to 1.0 = fused features close to CLIP baseline)')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Directory with _f.npy and _f_dino.npy files')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save fusion network weights')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--d_unified', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--loss_type', type=str, default='mse_cos',
                        choices=['mse_cos', 'mse_only', 'infonce'],
                        help='Loss function: mse_cos (orig), mse_only, infonce')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature (only for infonce)')
    parser.add_argument('--use_depth', action='store_true',
                        help='Enable third modality: depth statistics (_f_depth.npy)')
    parser.add_argument('--dim_depth', type=int, default=8,
                        help='Depth statistic dimension (only with --use_depth)')
    parser.add_argument('--depth_loss_weight', type=float, default=1.0,
                        help='Weight of auxiliary depth reconstruction loss (only with --use_depth)')
    args = parser.parse_args()

    train_fusion(
        feat_dir=args.feat_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        d_unified=args.d_unified,
        n_heads=args.n_heads,
        loss_type=args.loss_type,
        temperature=args.temperature,
        use_depth=args.use_depth,
        dim_depth=args.dim_depth,
        depth_loss_weight=args.depth_loss_weight,
    )
