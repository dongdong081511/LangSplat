"""Merge CLIP and DINO encoded features into a dual-stream directory.

Concatenates per-frame tile features: clip (N1d) + dino (N2d) -> (N1+N2)d.
Seg maps copied from CLIP dir.

Usage:
    python -m mm_langsplat.merge_dual \
        --dataset_path dataset/lerf_ovs/teatime \
        --clip_subdir language_features_dim8 \
        --dino_subdir language_features_dim8_dino \
        --out_subdir language_features_dim16_dual
"""
import os
import sys
import shutil
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--clip_subdir', type=str, required=True)
    parser.add_argument('--dino_subdir', type=str, required=True)
    parser.add_argument('--out_subdir', type=str, required=True)
    args = parser.parse_args()

    clip_dir = os.path.join(args.dataset_path, args.clip_subdir)
    dino_dir = os.path.join(args.dataset_path, args.dino_subdir)
    out_dir = os.path.join(args.dataset_path, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    feat_files = sorted(f for f in os.listdir(clip_dir) if f.endswith('_f.npy'))
    for f in tqdm(feat_files):
        clip_feat = np.load(os.path.join(clip_dir, f))
        dino_f = f  # same naming in dino dir (produced by encode_dim3)
        dino_feat = np.load(os.path.join(dino_dir, dino_f))
        assert clip_feat.shape[0] == dino_feat.shape[0], \
            f"row mismatch {f}: {clip_feat.shape[0]} vs {dino_feat.shape[0]}"
        merged = np.concatenate([clip_feat, dino_feat], axis=1)
        np.save(os.path.join(out_dir, f), merged)
        seg = f.replace('_f.npy', '_s.npy')
        seg_src = os.path.join(clip_dir, seg)
        if os.path.exists(seg_src):
            shutil.copy(seg_src, os.path.join(out_dir, seg))

    sample_c = np.load(os.path.join(clip_dir, feat_files[0]))
    sample_m = np.load(os.path.join(out_dir, feat_files[0]))
    print(f"merged {len(feat_files)} frames: {sample_c.shape} -> {sample_m.shape}")
    print(f"saved to {out_dir}")


if __name__ == '__main__':
    main()
