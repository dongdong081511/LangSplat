#!/usr/bin/env python
# EXP-027 gate: masked tile CLIP pooling (bbox background suppression) vs baseline tile features.
# Rebuilds per-tile pixel masks from _s.npy (no SAM rerun), re-encodes each tile with
# background pixels filled, then evaluates with the exact tile protocol (dense_upper_test).
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from openclip_encoder import OpenCLIPNetwork
from dense_upper_test import eval_frame

GRAY = (0.48145466, 0.4578275, 0.40821073)


def pca_compress(feats, dim):
    """[T,D] normalized -> PCA project to dim -> renormalize (AE-rate simulator)."""
    X = feats - feats.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:dim].T @ Vt[:dim]
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return P.astype(np.float32)


def encode_tiles_masked(img01, s, clip_model, mode):
    """img01: [H,W,3] RGB 0-1; s: [4,H,W] global tile ids -> [T,512] np normalized."""
    T = int(s.max()) + 1
    out = np.zeros((T, clip_model.clip_n_dims), np.float32)
    done = np.zeros(T, bool)
    for li in range(s.shape[0]):
        sm = s[li].astype(np.int64)
        for t in np.unique(sm):
            if t < 0 or done[t]:
                continue
            region = sm == t
            ys, xs = np.where(region)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = img01[y0:y1, x0:x1].copy()
            m = region[y0:y1, x0:x1]
            if mode == 'gray':
                crop[~m] = GRAY
            elif mode == 'black':
                crop[~m] = 0.0
            elif mode == 'blur':
                k = max(3, int(min(crop.shape[:2]) * 0.15) | 1)
                crop[~m] = cv2.GaussianBlur(crop, (k, k), 0)[~m]
            tt = torch.from_numpy(crop).permute(2, 0, 1)[None].float()
            with torch.no_grad():
                e = clip_model.model.encode_image(clip_model.process(tt).half().cuda())
            e = e[0].float().cpu().numpy()
            out[t] = e / np.linalg.norm(e)
            done[t] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default='dataset/lerf_ovs/teatime')
    ap.add_argument('--json_folder', default='dataset/lerf_ovs/label/teatime')
    ap.add_argument('--max_frames', type=int, default=100)
    ap.add_argument('--modes', nargs='+', default=['none', 'gray', 'black', 'blur'])
    ap.add_argument('--clip_model', default='ViT-B-16')
    ap.add_argument('--pretrained', default='laion2b_s34b_b88k')
    ap.add_argument('--clip_n_dims', type=int, default=512)
    ap.add_argument('--pca_dim', type=int, default=0, help='if >0, PCA-compress feats to this dim (AE-rate simulator)')
    args = ap.parse_args()

    cm = OpenCLIPNetwork('cuda', clip_model_type=args.clip_model,
                         clip_model_pretrained=args.pretrained, clip_n_dims=args.clip_n_dims)
    cm.use_templates = False

    jsons = sorted(glob.glob(os.path.join(args.json_folder, 'frame_*.json')))[:args.max_frames]
    frames = []
    for jf in jsons:
        stem = os.path.basename(jf).replace('.json', '')
        info = json.load(open(jf))
        h, w = info['info']['height'], info['info']['width']
        img = cv2.imread(os.path.join(args.dataset_path, 'images', stem + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        s = np.load(os.path.join(args.dataset_path, 'language_features', stem + '_s.npy'))
        gt_masks = {}
        for o in info['objects']:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.asarray(o['segmentation'], np.int32)], 1)
            gt_masks[o['category']] = np.maximum(gt_masks.get(o['category'], np.zeros_like(m)), m)
        frames.append((stem, img, s, gt_masks, h, w))
    print(f'{len(frames)} frames | modes={args.modes}', flush=True)

    for mode in args.modes:
        feats = {}
        for stem, img, s, _, _, _ in frames:
            feats[stem] = encode_tiles_masked(img, s, cm, mode)
            print(f'  encoded [{mode}] {stem}', flush=True)
        if args.pca_dim > 0:
            for stem in feats:
                feats[stem] = pca_compress(feats[stem], args.pca_dim)
            print(f'  PCA -> {args.pca_dim}d applied', flush=True)
        for use_t in [False, True]:
            cm.use_templates = use_t
            ious_all = []
            for stem, _, s, gt_masks, h, w in frames:
                res = eval_frame(feats[stem], s, gt_masks, h, w, cm)
                ious_all.extend(v[0] for v in res.values())
            tag = ' +templates' if use_t else ''
            print(f'>>> [masked-pool {mode}{tag}] mIoU chosen-level = {np.mean(ious_all):.4f}  (n={len(ious_all)})',
                  flush=True)


if __name__ == '__main__':
    main()
