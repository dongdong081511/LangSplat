#!/usr/bin/env python
# 2D single-frame relevancy upper-bound test: global vs dense tile features.
# Protocol EXACTLY mirrors evaluate_iou_loc.activate_stream:
#   tile relevancy map -> 30x30 avg filter -> 0.5*(avg+raw) -> minmax -> *2-1 -> clip -> >0.5
#   -> majority smooth -> IoU; chosen level = argmax of filtered map peak (like real eval).
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


def smooth(mask):
    """Exact vectorized replica of eval/utils.smooth (7x7 majority, clipped window,
    including its exclusive h-1/w-1 upper bound and tie->0 behavior)."""
    h, w = mask.shape[:2]
    m = mask.astype(np.int32)
    S = np.zeros((h + 1, w + 1), dtype=np.int64)
    S[1:, 1:] = m.cumsum(0).cumsum(1)
    ii = np.arange(h)[:, None]
    jj = np.arange(w)[None, :]
    r1 = np.maximum(ii - 3, 0); r2 = np.minimum(ii + 4, h - 1)
    c1 = np.maximum(jj - 3, 0); c2 = np.minimum(jj + 4, w - 1)
    cnt = S[r2, c2] - S[r1, c2] - S[r2, c1] + S[r1, c1]
    area = (r2 - r1) * (c2 - c1)
    return (2 * cnt > area).astype(np.uint8)


def relev_tile_scores(emb, pos_e, neg_e, temp=10.0):
    """emb: [T,512] normalized fp32 cuda; pos_e: [1,512]; neg_e: [N,512]. Returns [T]."""
    sims_p = emb @ pos_e.T                      # [T,1]
    sims_n = emb @ neg_e.T                      # [T,N]
    pair = torch.stack((sims_p.repeat(1, sims_n.shape[1]), sims_n), dim=-1)
    sm = torch.softmax(temp * pair, dim=-1)
    return sm[..., 0][torch.arange(emb.shape[0]), sm[..., 0].argmin(dim=1)]  # hardest neg, like eval


def tile_to_map(relev, seg_i, h, w):
    seg = seg_i.astype(np.int64)
    rmap = relev[np.clip(seg, 0, None)].reshape(h, w).astype(np.float32)
    rmap[seg < 0] = 0.0
    return rmap


def eval_frame(feat, seg_maps, gt_masks, h, w, clip_model):
    """feat: [T,512]; seg_maps: [4,H,W]; gt_masks: {prompt: [H,W] uint8}.
    Returns {prompt: (chosen_iou, chosen_lvl, per_level_iou[4])} following activate_stream."""
    import torch.nn.functional as F  # noqa: F401
    emb = torch.from_numpy(feat).float().cuda()
    neg_e = clip_model.neg_embeds.float()
    k30 = np.ones((30, 30)) / 900
    out = {}
    for prompt, gt in gt_masks.items():
        clip_model.set_positives([prompt], use_templates=clip_model.use_templates)
        relev = relev_tile_scores(emb, clip_model.pos_embeds.float(), neg_e).cpu().numpy()
        iou_lvl = np.zeros(seg_maps.shape[0])
        score_lvl = np.zeros(seg_maps.shape[0])
        mask_lvl = []
        for li in range(seg_maps.shape[0]):
            rmap = tile_to_map(relev, seg_maps[li], h, w)
            filt = cv2.filter2D(rmap, -1, k30)
            comb = 0.5 * (filt + rmap)
            score_lvl[li] = comb.max()
            o = comb - comb.min()
            o = o / (o.max() + 1e-9)
            o = np.clip(o * 2.0 - 1.0, 0, 1)
            mask_pred = smooth((o > 0.5).astype(np.uint8))
            mask_lvl.append(mask_pred)
            inter = np.logical_and(gt, mask_pred).sum()
            union = np.logical_or(gt, mask_pred).sum()
            iou_lvl[li] = inter / (union + 1e-9)
        chosen = int(np.argmax(score_lvl))
        out[prompt] = (float(iou_lvl[chosen]), chosen, iou_lvl.tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default='dataset/lerf_ovs/teatime')
    ap.add_argument('--subdirs', nargs='+', default=['language_features', 'language_features_dense'])
    ap.add_argument('--json_folder', default='dataset/lerf_ovs/label/teatime')
    ap.add_argument('--max_frames', type=int, default=100)
    ap.add_argument('--use_templates', action='store_true')
    args = ap.parse_args()

    clip_model = OpenCLIPNetwork('cuda')
    clip_model.use_templates = args.use_templates

    jsons = sorted(glob.glob(os.path.join(args.json_folder, 'frame_*.json')))[:args.max_frames]
    print(f'evaluating {len(jsons)} frames from {args.json_folder}', flush=True)

    for subdir in args.subdirs:
        chosen_all = []
        per_level = [[], [], [], []]
        for jf in jsons:
            stem = os.path.basename(jf).replace('.json', '')
            f = np.load(os.path.join(args.dataset_path, subdir, stem + '_f.npy')).astype(np.float32)
            s = np.load(os.path.join(args.dataset_path, subdir, stem + '_s.npy'))
            info = json.load(open(jf))
            h, w = info['info']['height'], info['info']['width']
            gt_masks = {}
            for o in info['objects']:
                pts = np.asarray(o['segmentation'], dtype=np.int32)
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [pts], 1)
                gt_masks[o['category']] = np.maximum(gt_masks.get(o['category'], np.zeros_like(m)), m)
            res = eval_frame(f, s, gt_masks, h, w, clip_model)
            for p, (ci, cl, lvl) in res.items():
                chosen_all.append(ci)
                for li, v in enumerate(lvl):
                    per_level[li].append(v)
                print(f'  [{subdir}] {stem} {p:18s} chosen=L{cl} IoU={ci:.4f}  levels=' +
                      ' '.join(f'{v:.3f}' for v in lvl), flush=True)
        lvl_str = ' '.join(f'L{i}={np.mean(v):.4f}' for i, v in enumerate(per_level))
        tag = ' +templates' if args.use_templates else ''
        print(f'>>> [{subdir}{tag}] mIoU chosen-level = {np.mean(chosen_all):.4f} | {lvl_str}', flush=True)


if __name__ == '__main__':
    main()
