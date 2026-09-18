"""Generate blended CLIP+DINOv2 features (no fusion training needed).

Projects DINOv2 (768d) to 512d via PCA, then blends with CLIP:
    fused = alpha * CLIP + (1-alpha) * PCA(DINOv2), then normalize

This is a fixed (non-learned) fusion that enriches CLIP with DINOv2 info.
The blend ratio alpha controls how much DINOv2 is added.

Usage:
    python -m mm_langsplat.gen_blend_features \
        --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
        --out_dir dataset/lerf_ovs/teatime/language_features_blend02 \
        --alpha 0.8
"""
import os
import sys
import shutil
import argparse
import glob
import torch
import numpy as np
from tqdm import tqdm


def fit_pca_projection(dino_feats, target_dim=512):
    """Fit PCA on DINOv2 features using numpy SVD.

    Args:
        dino_feats: [N, 768] DINOv2 features
        target_dim: 512

    Returns:
        proj_matrix: [768, 512] projection matrix (top components)
        mean: [768] mean
    """
    print(f'Fitting PCA (numpy SVD): {dino_feats.shape} -> {target_dim}d')
    mean = dino_feats.mean(axis=0)  # [768]
    centered = dino_feats - mean  # [N, 768]

    # SVD: centered = U @ S @ Vt
    # Vt: [768, 768] (or [min(N,768), 768])
    # Top target_dim components: Vt[:target_dim, :].T = [768, target_dim]
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_matrix = Vt[:target_dim, :].T  # [768, target_dim]

    # Explained variance
    total_var = (S ** 2).sum()
    explained_var = (S[:target_dim] ** 2).sum() / total_var
    print(f'PCA explained variance: {explained_var:.4f}')
    return proj_matrix, mean


def generate_blend(feat_dir, out_dir, alpha=0.8):
    """Generate blended features.

    Args:
        feat_dir: directory with _f.npy (CLIP) and _f_dino.npy (DINOv2)
        out_dir: output directory
        alpha: blend ratio (1.0=pure CLIP, 0.0=pure DINOv2)
    """
    os.makedirs(out_dir, exist_ok=True)

    # First pass: collect all DINOv2 features for PCA fitting
    f_files = sorted(glob.glob(os.path.join(feat_dir, '*_f.npy')))
    f_files = [f for f in f_files if not f.endswith('_dino.npy')]

    print(f'Collecting DINOv2 features from {len(f_files)} images for PCA...')
    dino_list = []
    for f_path in f_files:
        dino_path = f_path.replace('_f.npy', '_f_dino.npy')
        if not os.path.exists(dino_path):
            continue
        dino_feat = np.load(dino_path)
        clip_feat = np.load(f_path)
        valid = np.linalg.norm(clip_feat, axis=1) > 0.1
        dino_list.append(dino_feat[valid])

    all_dino = np.concatenate(dino_list, axis=0)
    print(f'Total DINOv2 features: {all_dino.shape}')

    # Fit PCA
    proj_matrix, mean = fit_pca_projection(all_dino, target_dim=512)

    # Second pass: blend and save
    print(f'Blending with alpha={alpha} (CLIP={alpha}, DINOv2={1-alpha})...')
    total_cos = []
    for f_path in tqdm(f_files, desc='Blending'):
        basename = os.path.basename(f_path).replace('_f.npy', '')
        dino_path = os.path.join(feat_dir, basename + '_f_dino.npy')
        s_path = os.path.join(feat_dir, basename + '_s.npy')

        if not os.path.exists(dino_path):
            shutil.copy(f_path, os.path.join(out_dir, basename + '_f.npy'))
            shutil.copy(s_path, os.path.join(out_dir, basename + '_s.npy'))
            continue

        clip_feat = np.load(f_path)  # [num_tiles, 512]
        dino_feat = np.load(dino_path)  # [num_tiles, 768]

        valid = np.linalg.norm(clip_feat, axis=1) > 0.1
        clip_valid = clip_feat[valid]  # [n, 512]
        dino_valid = dino_feat[valid]  # [n, 768]

        # Project DINOv2 to 512
        dino_proj = (dino_valid - mean) @ proj_matrix  # [n, 512]
        dino_proj = dino_proj / (np.linalg.norm(dino_proj, axis=1, keepdims=True) + 1e-8)

        # Blend
        blended = alpha * clip_valid + (1 - alpha) * dino_proj
        blended = blended / (np.linalg.norm(blended, axis=1, keepdims=True) + 1e-8)

        # Track similarity to CLIP
        cos_sim = (blended * clip_valid).sum(axis=1).mean()
        total_cos.append(cos_sim)

        # Save in same format (pad back to original size)
        fused_full = np.zeros_like(clip_feat)
        fused_full[valid] = blended.astype(np.float16)
        np.save(os.path.join(out_dir, basename + '_f.npy'), fused_full)
        shutil.copy(s_path, os.path.join(out_dir, basename + '_s.npy'))

    print(f'Avg cos_sim(blend, CLIP): {np.mean(total_cos):.4f}')
    print(f'Fused features saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Blend ratio: 1.0=pure CLIP, 0.0=pure DINOv2')
    args = parser.parse_args()

    generate_blend(args.feat_dir, args.out_dir, args.alpha)
