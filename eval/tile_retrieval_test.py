"""Task-routing feasibility test: image-query tile retrieval, CLIP vs DINOv2.

Protocols (no training, pure feature-space):
  intra-frame : query tile from GT bbox -> retrieve among same-frame tiles (exclude self)
  cross-frame : query tile from frame A bbox -> retrieve among frame B tiles,
                hit if top-1 tile center falls inside frame B's same-label GT bbox

A retrieval hit rate advantage for DINO over CLIP would justify building a
dedicated DINO feature field for image-query localization (task routing).
"""
import json
import glob
import os
import numpy as np
import cv2

ROOT = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs'
SCENE = 'teatime'
LABEL_DIR = os.path.join(ROOT, 'label', SCENE)
SOURCES = {
    'CLIP': os.path.join(ROOT, SCENE, 'language_features'),
    'DINO': os.path.join(ROOT, SCENE, 'language_features_dino'),
}
LEVEL_NAMES = ['default', 's', 'm', 'l']


def load_gt():
    gt = {}
    for js in sorted(glob.glob(os.path.join(LABEL_DIR, 'frame_*.json'))):
        with open(js) as f:
            d = json.load(f)
        h, w = d['info']['height'], d['info']['width']
        frame = d['info']['name'].split('.jpg')[0]
        objs = []
        for o in d['objects']:
            mask = np.zeros((h, w), np.uint8)
            pts = np.array(o['segmentation'], np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
            x1, y1, x2, y2 = np.array(o['bbox']).reshape(-1)
            objs.append({'label': o['category'], 'mask': mask,
                         'bbox': (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))})
        gt[frame] = objs
    return gt


def tile_centers(seg):
    """seg: [H, W] int -> dict tile_idx -> (cy, cx)."""
    idxs = np.unique(seg)
    idxs = idxs[idxs >= 0]
    out = {}
    for t in idxs:
        ys, xs = np.nonzero(seg == t)
        out[int(t)] = (ys.mean(), xs.mean())
    return out


def pick_query_tile(seg, obj_mask):
    """Largest-overlap tile whose pixels overlap GT mask; return tile idx or None."""
    idxs = np.unique(seg[obj_mask > 0])
    idxs = idxs[idxs >= 0]
    if len(idxs) == 0:
        return None
    counts = [(int(t), int((seg[obj_mask > 0] == t).sum())) for t in idxs]
    counts.sort(key=lambda x: -x[1])
    return counts[0][0]


def point_in_bbox(pt, bbox):
    y, x = pt
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def run(src_dir, gt, tag):
    feats, segs, centers = {}, {}, {}
    for frame in gt:
        f = np.load(os.path.join(src_dir, f'{frame}_f.npy')).astype(np.float32)
        s = np.load(os.path.join(src_dir, f'{frame}_s.npy')).astype(np.int64)
        feats[frame], segs[frame] = f, s
        centers[frame] = [tile_centers(s[i]) for i in range(4)]

    stats = {}  # (level, proto) -> [n, hit1, hit5]
    for frame, objs in gt.items():
        for li in range(4):
            seg = segs[frame][li]
            cent = centers[frame][li]
            F = feats[frame]
            for obj in objs:
                q = pick_query_tile(seg, obj['mask'])
                if q is None:
                    continue
                qv = F[q]
                sims = F @ qv
                order = np.argsort(-sims)
                for proto in ('intra', 'cross'):
                    if proto == 'intra':
                        cand = [t for t in order if t != q and t in cent]
                        hit_box, hit_list = [obj['bbox']], None
                    else:
                        other = [fr for fr in gt if fr != frame]
                        cand, hit_box = [], None
                        for fr2 in other:
                            match = [o for o in gt[fr2] if o['label'] == obj['label']]
                            if not match:
                                continue
                            li2 = li
                            cent2 = centers[fr2][li2]
                            cand = [(fr2, t) for t in np.argsort(
                                -(feats[fr2] @ qv)) if t in cent2]
                            hit_box = [match[0]['bbox']]
                            break
                        if hit_box is None:
                            continue
                    top5 = cand[:5] if proto == 'intra' else cand[:5]
                    k = (li, proto)
                    st = stats.setdefault(k, [0, 0, 0])
                    st[0] += 1
                    if top5:
                        if proto == 'intra':
                            if point_in_bbox(cent[top5[0]], hit_box[0]):
                                st[1] += 1
                            if any(point_in_bbox(cent[t], hit_box[0]) for t in top5):
                                st[2] += 1
                        else:
                            fr2 = top5[0][0]
                            if point_in_bbox(centers[fr2][li][top5[0][1]], hit_box[0]):
                                st[1] += 1
                            if any(point_in_bbox(centers[t[0]][li][t[1]], hit_box[0]) for t in top5):
                                st[2] += 1
    print(f'\n=== {tag} ===')
    for (li, proto), (n, h1, h5) in sorted(stats.items()):
        print(f'  {LEVEL_NAMES[li]:8s} {proto:6s}  n={n:3d}  top1={h1}/{n}={h1/n:.2%}  top5={h5}/{n}={h5/n:.2%}')
    tot = {}
    for (li, proto), (n, h1, h5) in stats.items():
        t = tot.setdefault(proto, [0, 0, 0])
        t[0] += n; t[1] += h1; t[2] += h5
    for proto, (n, h1, h5) in sorted(tot.items()):
        print(f'  TOTAL {proto:6s}  n={n}  top1={h1/n:.2%}  top5={h5/n:.2%}')


if __name__ == '__main__':
    gt = load_gt()
    n_obj = sum(len(v) for v in gt.values())
    print(f'GT frames: {sorted(gt.keys())}, objects: {n_obj}')
    for tag, src in SOURCES.items():
        run(src, gt, tag)
