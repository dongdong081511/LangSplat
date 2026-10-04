"""Extract per-tile depth statistics from Depth Anything V2 full-image depth maps.

For each frame:
  1. Run DepthAnythingV2 (vitl) on the full image -> relative depth [H, W],
     normalized per-image to [0, 1] (0=near, 1=far).
  2. For each SAM tile id in the existing seg maps (_s.npy), collect the depth
     values inside the tile region and compute an 8-dim statistic vector:
     [mean, std, min, max, p25, p75, grad_mean, grad_max]
  3. Save as <frame>_f_depth.npy [max_tiles, 8] in the feature directory
     (zero-padded, same layout as _f.npy / _f_dino.npy).

Reuses existing seg maps so SAM does not need to be re-run.

Usage:
    python -u -m mm_langsplat.extract_depth_tiles \
        --scene_dir dataset/lerf_ovs/teatime \
        --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
        --depth_ckpt ckpts/depth_anything_v2_vitl.pth
"""
import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_image_resized(image_path):
    """Mirror preprocess.py image loading (resize >1080p height)."""
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    if orig_h > 1080:
        scale = orig_h / 1080
    else:
        scale = 1.0
    new_size = (int(orig_w / scale), int(orig_h / scale))
    return cv2.resize(image, new_size), new_size


def tile_depth_stats(depth, tile_mask):
    """Compute 8-dim depth statistics inside a tile region.

    Args:
        depth: [H, W] float32 normalized depth map
        tile_mask: [H, W] bool mask of the tile region
    Returns:
        [8] float32: mean, std, min, max, p25, p75, grad_mean, grad_max
    """
    vals = depth[tile_mask]
    if vals.size == 0:
        return np.zeros(8, dtype=np.float32)
    gy, gx = np.gradient(depth)
    grad = np.sqrt(gy ** 2 + gx ** 2)[tile_mask]
    return np.array([
        vals.mean(),
        vals.std(),
        vals.min(),
        vals.max(),
        np.percentile(vals, 25),
        np.percentile(vals, 75),
        grad.mean(),
        grad.max(),
    ], dtype=np.float32)


def extract_scene(scene_dir, feat_dir, depth_ckpt, device='cuda'):
    repo_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'third_party', 'Depth-Anything-V2')
    sys.path.insert(0, repo_path)
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
        use_bn=False, use_clstoken=False,
    )
    model.load_state_dict(torch.load(depth_ckpt, map_location='cpu'))
    model.to(device).eval()

    image_dir = os.path.join(scene_dir, 'images')
    s_files = sorted(glob.glob(os.path.join(feat_dir, '*_s.npy')))
    print(f'{len(s_files)} frames to process')

    for s_path in tqdm(s_files, desc='Depth tiles'):
        basename = os.path.basename(s_path).replace('_s.npy', '')
        out_path = os.path.join(feat_dir, basename + '_f_depth.npy')
        if os.path.exists(out_path):
            continue

        img_bgr, (w, h) = load_image_resized(
            os.path.join(image_dir, basename + '.jpg'))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth = model.infer_image(img_rgb, input_size=518)  # [H, W]
        depth = depth.astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-6)

        seg_map = np.load(s_path)  # [C, H, W] tile ids (cumulative), -1 = none
        # CLIP _f.npy has exactly seg_map.max()+1 rows (asserted in preprocess.py)
        num_tiles = int(seg_map.max()) + 1

        feats = np.zeros((num_tiles, 8), dtype=np.float32)
        done = set()
        for c in range(seg_map.shape[0]):
            channel = seg_map[c]
            for tile_id in np.unique(channel):
                tid = int(tile_id)
                if tid < 0 or tid >= feats.shape[0] or tid in done:
                    continue  # tile already covered by an earlier channel
                done.add(tid)
                mask = channel == tile_id
                feats[tid] = tile_depth_stats(depth, mask)

        np.save(out_path, feats)

    print(f'Done. Saved *_f_depth.npy to {feat_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Scene dir containing images/')
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Feature dir with _s.npy files (output goes here)')
    parser.add_argument('--depth_ckpt', type=str,
                        default='ckpts/depth_anything_v2_vitl.pth')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    extract_scene(args.scene_dir, args.feat_dir, args.depth_ckpt, args.device)
