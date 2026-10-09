#!/usr/bin/env python
"""EXP-061: 按 GT 类别分组量测跨帧特征一致性 (DINO vs CLIP).

机理预测 (EXP-060 后的视角方差假说):
  金属反光类 (knife): DINO 跨帧 cos 显著低于 CLIP (反光逐帧剧变, CLIP 语义恒定)
  非金属类: 差距缩小

用法: cd eval && python analyze_view_consistency.py --scene waldo_kitchen
"""
import os, sys, json, argparse, itertools
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_image_query import load_gt, pick_query_tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='waldo_kitchen')
    ap.add_argument('--scene_root', default='../dataset/lerf_ovs')
    ap.add_argument('--gt_dir', default='')
    ap.add_argument('--dino_subdir', default='language_features_dim64_dino_native')
    ap.add_argument('--clip_subdir', default='language_features_dim8')
    ap.add_argument('--seg_index', type=int, default=0)
    args = ap.parse_args()

    gt_dir = args.gt_dir or os.path.join(args.scene_root, args.scene, 'label')
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join('../dataset/lerf_ovs/label', args.scene)
    gt = load_gt(gt_dir)
    frames = sorted(gt.keys())
    print(f'scene={args.scene} GT frames={len(frames)} objects={sum(len(v) for v in gt.values())}')

    # 每场收集: label -> list of (frame, feat_dino_norm, feat_clip_norm)
    def load_field(subdir):
        per_label = {}
        for fr in frames:
            F = np.load(os.path.join(args.scene_root, args.scene, subdir, f'{fr}_f.npy')).astype(np.float32)
            S = np.load(os.path.join(args.scene_root, args.scene, subdir, f'{fr}_s.npy')).astype(np.int64)[args.seg_index]
            for obj in gt[fr]:
                q = pick_query_tile(S, obj['mask'])
                if q is None or q >= F.shape[0]:
                    continue
                v = F[q]
                v = v / (np.linalg.norm(v) + 1e-8)
                per_label.setdefault(obj['label'], []).append((fr, v))
        return per_label

    dino = load_field(args.dino_subdir)
    clip = load_field(args.clip_subdir)
    labels = sorted(set(dino) & set(clip))

    def crossframe_cos(entries):
        cs = []
        for (fa, va), (fb, vb) in itertools.combinations(entries, 2):
            if fa == fb:
                continue
            cs.append(float(va @ vb))
        return cs

    metal_kw = ('knife', 'blade', 'scissor', 'foil', 'pot', 'pan', 'kettle', 'faucet')
    rows = []
    for lb in labels:
        cd = crossframe_cos(dino[lb])
        cc = crossframe_cos(clip[lb])
        if not cd or not cc:
            continue
        is_metal = any(k in lb.lower() for k in metal_kw)
        rows.append((lb, len(dino[lb]), len(cd), np.mean(cd), np.mean(cc), is_metal))

    rows.sort(key=lambda r: (r[3] - r[4]))  # dino-clip 差距最小的(受伤害最大)在前
    print(f'\n{"label":<20} {"n_inst":>6} {"n_pair":>6} {"dino_cos":>9} {"clip_cos":>9} {"Δ(D-C)":>8} metal')
    for lb, ni, npair, md, mc, is_m in rows:
        print(f'{lb:<20} {ni:>6} {npair:>6} {md:>9.4f} {mc:>9.4f} {md - mc:>+8.4f} {"*" if is_m else ""}')

    for name, sel in [('metal/reflective', [r for r in rows if r[5]]),
                      ('non-metal       ', [r for r in rows if not r[5]]),
                      ('ALL              ', rows)]:
        if not sel:
            continue
        md = np.mean([r[3] for r in sel]); mc = np.mean([r[4] for r in sel])
        w = sum(len(crossframe_cos(dino[r[0]])) for r in sel)
        print(f'\n[{name}] labels={len(sel)} pairs={w}  dino_cos={md:.4f}  clip_cos={mc:.4f}  Δ={md - mc:+.4f}')


if __name__ == '__main__':
    main()
