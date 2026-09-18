"""Generate fused 512-dim features using a trained fusion network.

Reads CLIP (_f.npy) and DINOv2 (_f_dino.npy) features, runs the fusion
network in inference mode, and saves fused features as _f.npy in a new
directory. Also copies seg maps (_s.npy) unchanged.

The output directory has the same format as the original language_features/,
so the downstream pipeline (AE training, encoding, gaussian, eval) works
unchanged.

Usage:
    python -m mm_langsplat.gen_fused_features \
        --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
        --out_dir dataset/lerf_ovs/teatime/language_features_fused \
        --ckpt_path mm_langsplat/ckpt/fusion_teatime.pth
"""
import os
import sys
import shutil
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm_langsplat.fusion.cross_attention import TileCrossAttentionFusion


def generate_fused(feat_dir, out_dir, ckpt_path):
    """Generate fused features using trained fusion network.

    Args:
        feat_dir: directory with _f.npy and _f_dino.npy
        out_dir: output directory for fused _f.npy (and copied _s.npy)
        ckpt_path: trained fusion network weights
    """
    device = torch.device('cuda')
    os.makedirs(out_dir, exist_ok=True)

    # Load fusion network
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']
    model = TileCrossAttentionFusion(
        dim_clip=config['dim_clip'],
        dim_dino=config['dim_dino'],
        d_unified=config['d_unified'],
        d_out=config['d_out'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded fusion network from {ckpt_path}')

    # Process each image
    f_files = sorted(glob.glob(os.path.join(feat_dir, '*_f.npy')))
    f_files = [f for f in f_files if not f.endswith('_dino.npy')]

    total_cos_sim = 0.0
    count = 0

    for f_path in tqdm(f_files, desc='Generating fused features'):
        basename = os.path.basename(f_path).replace('_f.npy', '')
        dino_path = os.path.join(feat_dir, basename + '_f_dino.npy')
        s_path = os.path.join(feat_dir, basename + '_s.npy')

        if not os.path.exists(dino_path):
            print(f'WARNING: DINO not found for {basename}, copying CLIP as-is')
            shutil.copy(f_path, os.path.join(out_dir, basename + '_f.npy'))
            shutil.copy(s_path, os.path.join(out_dir, basename + '_s.npy'))
            continue

        clip_feat = np.load(f_path)  # [num_tiles, 512]
        dino_feat = np.load(dino_path)  # [num_tiles, 768]

        clip_t = torch.from_numpy(clip_feat).float().to(device)
        dino_t = torch.from_numpy(dino_feat).float().to(device)

        # Run fusion (inference, no dropout)
        with torch.no_grad():
            fused = model(clip_t, dino_t, is_training=False)
            # Track similarity to CLIP for diagnostics
            if count < 5:
                cos_sim = F.cosine_similarity(fused, clip_t, dim=-1).mean()
                total_cos_sim += cos_sim.item()
                count += 1

        # Save fused features
        fused_np = fused.cpu().numpy().astype(np.float32)
        np.save(os.path.join(out_dir, basename + '_f.npy'), fused_np)

        # Copy seg maps unchanged
        shutil.copy(s_path, os.path.join(out_dir, basename + '_s.npy'))

    if count > 0:
        print(f'Avg cos_sim (fused vs CLIP) for first {count} images: {total_cos_sim/count:.4f}')
    print(f'Fused features saved to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    generate_fused(args.feat_dir, args.out_dir, args.ckpt_path)
