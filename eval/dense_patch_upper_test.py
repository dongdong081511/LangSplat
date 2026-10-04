#!/usr/bin/env python
# EXP-026 gate test: dense per-patch CLIP (MaskCLIP) 2D upper bound vs tile baseline.
# Teachers: (a) single-resize 14x14 patch map, (b) sliding-window dense map (224 crop, stride).
# Protocol mirrors activate_stream exactly (same thresholds as dense_upper_test.py).
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from openclip_encoder import OpenCLIPNetwork
from dense_upper_test import smooth

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def relev_scores(feats, pos_e, neg_e, temp=10.0):
    """feats: [N,512] normalized cuda; pos_e: [1,512]; neg_e: [N_neg,512] -> [N] hardest-neg relevancy."""
    sims_p = feats @ pos_e.T
    sims_n = feats @ neg_e.T
    pair = torch.stack((sims_p.repeat(1, sims_n.shape[1]), sims_n), dim=-1)
    sm = torch.softmax(temp * pair, dim=-1)
    return sm[..., 0][torch.arange(feats.shape[0]), sm[..., 0].argmin(dim=1)]


def eval_map(relmap, gt, k30):
    filt = cv2.filter2D(relmap, -1, k30)
    comb = 0.5 * (filt + relmap)
    o = comb - comb.min()
    o = o / (o.max() + 1e-9)
    o = np.clip(o * 2.0 - 1.0, 0, 1)
    pred = smooth((o > 0.5).astype(np.uint8))
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return inter / (union + 1e-9), float(comb.max())


def make_patch_encoder(clip_model):
    """MaskCLIP per-patch encoder: last-block v-projection, keep 14x14 patches (no mean)."""
    vit = clip_model.model.visual
    blocks = vit.transformer.resblocks
    attn = blocks[-1].attn
    d = attn.embed_dim
    if hasattr(attn, 'in_proj_weight') and attn.in_proj_weight is not None:
        Wv, bv = attn.in_proj_weight[2 * d:, :], attn.in_proj_bias[2 * d:]
    else:
        Wv, bv = attn.qkv.weight[2 * d:, :], attn.qkv.bias[2 * d:]

    def encode_patch(img01):
        """img01: np [H,W,3] RGB 0-1 -> np [196,512] normalized fp32."""
        t = torch.from_numpy(img01).permute(2, 0, 1)[None].float()
        if t.shape[-1] != 224:
            t = F.interpolate(t, size=(224, 224), mode='bicubic', align_corners=False)
        t = (t - MEAN) / STD
        with torch.no_grad():
            x = vit.conv1(t.half().cuda())
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = vit.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], 1) + vit.positional_embedding.to(x.dtype)
            x = vit.ln_pre(x).permute(1, 0, 2)
            for blk in blocks[:-1]:
                x = blk(x)
            v = F.linear(x, Wv, bv)
            f = vit.ln_post(v[1:]) @ vit.proj
        f = f[:, 0, :].float().cpu().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    return encode_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', default='dataset/lerf_ovs/teatime')
    ap.add_argument('--json_folder', default='dataset/lerf_ovs/label/teatime')
    ap.add_argument('--max_frames', type=int, default=100)
    ap.add_argument('--use_templates', action='store_true')
    ap.add_argument('--stride', type=int, default=112)
    args = ap.parse_args()

    cm = OpenCLIPNetwork('cuda')
    cm.use_templates = args.use_templates
    neg_e = cm.neg_embeds.float()
    k30 = np.ones((30, 30)) / 900
    enc = make_patch_encoder(cm)

    jsons = sorted(glob.glob(os.path.join(args.json_folder, 'frame_*.json')))[:args.max_frames]
    print(f'{len(jsons)} frames | stride={args.stride} | templates={args.use_templates}', flush=True)

    for mode in ['single', 'sliding']:
        ious_all = []
        for jf in jsons:
            stem = os.path.basename(jf).replace('.json', '')
            info = json.load(open(jf))
            h, w = info['info']['height'], info['info']['width']
            img = cv2.imread(os.path.join(args.dataset_path, 'images', stem + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_masks = {}
            for o in info['objects']:
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [np.asarray(o['segmentation'], np.int32)], 1)
                gt_masks[o['category']] = np.maximum(gt_masks.get(o['category'], np.zeros_like(m)), m)

            if mode == 'single':
                pf = enc(img).reshape(14, 14, 512)
                for prompt, gt in gt_masks.items():
                    cm.set_positives([prompt], use_templates=args.use_templates)
                    rel = relev_scores(torch.from_numpy(pf.reshape(-1, 512)).float().cuda(),
                                       cm.pos_embeds.float(), neg_e).cpu().numpy().reshape(14, 14)
                    relmap = cv2.resize(rel.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                    iou, _ = eval_map(relmap, gt, k30)
                    ious_all.append(iou)
                    print(f'  [{mode}] {stem} {prompt:18s} IoU={iou:.4f}', flush=True)
            else:
                acc = np.zeros((h, w, 512), np.float32)
                cnt = np.zeros((h, w, 1), np.float32)
                ys = sorted(set(list(range(0, h - 223, args.stride)) + [h - 224]))
                xs = sorted(set(list(range(0, w - 223, args.stride)) + [w - 224]))
                for y0 in ys:
                    for x0 in xs:
                        pf = enc(img[y0:y0 + 224, x0:x0 + 224]).reshape(14, 14, 512)
                        acc[y0:y0 + 224, x0:x0 + 224] += np.repeat(np.repeat(pf, 16, 0), 16, 1)
                        cnt[y0:y0 + 224, x0:x0 + 224] += 1
                fmap = acc / cnt
                fmap = fmap / np.linalg.norm(fmap, axis=2, keepdims=True)
                fmap_t = torch.from_numpy(fmap.reshape(-1, 512)).float().cuda()
                for prompt, gt in gt_masks.items():
                    cm.set_positives([prompt], use_templates=args.use_templates)
                    rel = relev_scores(fmap_t, cm.pos_embeds.float(), neg_e).cpu().numpy().reshape(h, w)
                    iou, _ = eval_map(rel, gt, k30)
                    ious_all.append(iou)
                    print(f'  [{mode}] {stem} {prompt:18s} IoU={iou:.4f}', flush=True)

        tag = ' +templates' if args.use_templates else ''
        print(f'>>> [dense-patch {mode}{tag}] mIoU = {np.mean(ious_all):.4f}  (n={len(ious_all)})', flush=True)

    # discriminability diagnostic on first frame, first category
    jf = jsons[0]
    stem = os.path.basename(jf).replace('.json', '')
    info = json.load(open(jf))
    h, w = info['info']['height'], info['info']['width']
    img = cv2.imread(os.path.join(args.dataset_path, 'images', stem + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    target = info['objects'][0]['category']
    cm.set_positives([target], use_templates=args.use_templates)
    pe = cm.pos_embeds.float()
    for name, feats in [('single 196 patches', enc(img)),
                        ('tile baseline  [ref]', None)]:
        if feats is None:
            f = np.load(os.path.join(args.dataset_path, 'language_features', stem + '_f.npy')).astype(np.float32)
        else:
            f = feats
        ft = torch.from_numpy(f).float().cuda()
        ps = (ft @ pe.T).squeeze().cpu().numpy()
        rel = relev_scores(ft, pe, neg_e).cpu().numpy()
        print(f'[diag {stem} "{target}"] {name}: pos_sim std={ps.std():.4f} '
              f'relev p95-p5={np.percentile(rel, 95) - np.percentile(rel, 5):.4f}', flush=True)


if __name__ == '__main__':
    main()
