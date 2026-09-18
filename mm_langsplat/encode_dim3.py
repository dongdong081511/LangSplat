"""Encode 512-dim features to 3-dim using a trained autoencoder.

Generic version of encode_bed_dim3.py - works with any scene and AE checkpoint.

Usage:
    python -m mm_langsplat.encode_dim3 \
        --dataset_path ../dataset/lerf_ovs/teatime \
        --data_subdir language_features_blend02 \
        --out_subdir language_features_dim3_blend02 \
        --ae_ckpt autoencoder/ckpt/teatime_blend02/best_ckpt.pth
"""
import os
import sys
import shutil
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoencoder.model import Autoencoder


def encode_features(dataset_path, data_subdir, out_subdir, ae_ckpt,
                    encoder_dims=[256, 128, 64, 32, 3],
                    decoder_dims=[16, 32, 64, 128, 256, 256, 512]):
    """Encode 512-dim features to 3-dim using trained AE.

    Args:
        dataset_path: path to scene directory
        data_subdir: subdirectory with _f.npy (512-dim) and _s.npy
        out_subdir: output subdirectory for 3-dim features
        ae_ckpt: path to AE checkpoint
    """
    src_dir = os.path.join(dataset_path, data_subdir)
    dst_dir = os.path.join(dataset_path, out_subdir)
    os.makedirs(dst_dir, exist_ok=True)

    # Load AE
    ae = Autoencoder(encoder_dims, decoder_dims)
    ckpt = torch.load(ae_ckpt, map_location='cpu')
    ae.load_state_dict(ckpt)
    ae.eval()
    ae = ae.cuda()
    print(f'AE loaded from {ae_ckpt}')

    # Get feature files
    feat_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_f.npy')])
    print(f'Processing {len(feat_files)} files...')

    for feat_file in tqdm(feat_files):
        feat_path = os.path.join(src_dir, feat_file)
        feat = np.load(feat_path)

        with torch.no_grad():
            feat_tensor = torch.from_numpy(feat).float().cuda()
            dim3_feat = ae.encode(feat_tensor).cpu().numpy()

        np.save(os.path.join(dst_dir, feat_file), dim3_feat)

        # Copy seg map
        seg_file = feat_file.replace('_f.npy', '_s.npy')
        src_seg = os.path.join(src_dir, seg_file)
        if os.path.exists(src_seg):
            shutil.copy(src_seg, os.path.join(dst_dir, seg_file))

    # Verify
    print('\nVerification:')
    sample_feat = np.load(os.path.join(src_dir, feat_files[0]))
    sample_dim3 = np.load(os.path.join(dst_dir, feat_files[0]))
    with torch.no_grad():
        decoded = ae.decode(torch.from_numpy(sample_dim3).float().cuda()).cpu().numpy()

    orig_norm = sample_feat / (np.linalg.norm(sample_feat, axis=1, keepdims=True) + 1e-8)
    decoded_norm = decoded / (np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(orig_norm * decoded_norm, axis=1)
    print(f'cos_sim(orig, decode(encode(orig))): mean={cos_sims.mean():.4f}, '
          f'>0.9: {(cos_sims > 0.9).mean()*100:.1f}%')
    print(f'Done! Saved to {dst_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--data_subdir', type=str, required=True)
    parser.add_argument('--out_subdir', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--encoder_dims', nargs='+', type=int, default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 256, 512])
    args = parser.parse_args()

    encode_features(args.dataset_path, args.data_subdir, args.out_subdir, args.ae_ckpt,
                    args.encoder_dims, args.decoder_dims)
