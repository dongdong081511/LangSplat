"""Train a linear alignment W: DINOv2 (768) -> CLIP space (512).

Supervision: tile-level pairs (dino _f_dino.npy, clip _f.npy) from language_features_mm.
Loss: 1 - cosine(proj(dino), clip), clip targets L2-normalized.

Usage:
    python -m mm_langsplat.train_align \
        --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
        --save_path mm_langsplat/ckpt/align_teatime.pth
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AlignLayer(nn.Module):
    def __init__(self, dino_dim=768, clip_dim=512):
        super().__init__()
        self.proj = nn.Linear(dino_dim, clip_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='dir containing _f.npy (CLIP) and _f_dino.npy (DINOv2)')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dino_dim', type=int, default=768)
    parser.add_argument('--clip_dim', type=int, default=512)
    args = parser.parse_args()

    clip_feats, dino_feats = [], []
    files = sorted(f for f in os.listdir(args.feat_dir) if f.endswith('_f.npy'))
    for f in files:
        dino_f = f.replace('_f.npy', '_f_dino.npy')
        dino_path = os.path.join(args.feat_dir, dino_f)
        if not os.path.exists(dino_path):
            continue
        c = np.load(os.path.join(args.feat_dir, f))
        d = np.load(dino_path)
        assert c.shape[0] == d.shape[0], f"row mismatch {f}: {c.shape[0]} vs {d.shape[0]}"
        clip_feats.append(c)
        dino_feats.append(d)
    clip_all = torch.from_numpy(np.concatenate(clip_feats, 0)).float().cuda()
    dino_all = torch.from_numpy(np.concatenate(dino_feats, 0)).float().cuda()
    # normalize CLIP targets to unit sphere (cos loss); keep dino raw
    clip_n = F.normalize(clip_all, dim=-1)
    print(f"tiles: {clip_all.shape[0]}, clip {clip_all.shape[1]}d, dino {dino_all.shape[1]}d")

    torch.manual_seed(0)
    model = AlignLayer(args.dino_dim, args.clip_dim).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = clip_all.shape[0]
    bs = 256
    for ep in range(args.epochs):
        perm = torch.randperm(n, device='cuda')
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model(dino_all[idx])
            loss = (1 - F.cosine_similarity(pred, clip_n[idx], dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * idx.shape[0]
        if (ep + 1) % 10 == 0 or ep == 0:
            with torch.no_grad():
                pred_all = model(dino_all)
                cs = F.cosine_similarity(pred_all, clip_n, dim=-1)
            print(f"epoch {ep+1}: loss={tot/n:.4f} cos_sim={cs.mean().item():.4f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"saved: {args.save_path}")


if __name__ == '__main__':
    main()
