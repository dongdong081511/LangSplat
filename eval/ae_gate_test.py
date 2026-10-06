"""EXP-050 gate: 2D cross-frame tile retrieval in the AE code space.

Compares raw DINO features, the original AE-32d codes, and struct-loss AE codes
on the same discriminative task the 3D field must ultimately serve.  A struct
AE that beats or ties the original AE here has earned a 3D field run.

Protocol (mirrors tile_retrieval_test.py cross-frame):
  query = tile with max overlap of GT bbox (frame A, seg level 0)
  for every other frame B containing the same label: top-1 tile by cosine sim;
  hit if tile center falls in frame B's same-label bbox (any-bbox).
  Reports: frame-hit-rate (mean over frames, aligns with EXP-048 hit-rate)
           and chosen (best-scoring frame only, aligns with chosen-level).
"""
import json
import glob
import os
import sys
import numpy as np
import cv2

ROOT = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs'
SCENE = sys.argv[1] if len(sys.argv) > 1 else 'teatime'
DIRS = {
    'raw768': os.path.join(ROOT, SCENE, 'language_features_dino'),
    'ae32': os.path.join(ROOT, SCENE, 'language_features_dim32_dino'),
    'ae32_struct05': os.path.join(ROOT, SCENE, 'language_features_dim32_dino_struct05'),
}


def load_gt():
    gt = {}
    for js in sorted(glob.glob(os.path.join(ROOT, 'label', SCENE, 'frame_*.json'))):
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


def pick_query_tile(seg, obj_mask):
    idxs = np.unique(seg[obj_mask > 0])
    idxs = idxs[idxs >= 0]
    if len(idxs) == 0:
        return None
    counts = sorted(((int(t), int((seg[obj_mask > 0] == t).sum())) for t in idxs),
                    key=lambda x: -x[1])
    return counts[0][0]


def tile_center(seg, t):
    ys, xs = np.nonzero(seg == t)
    if len(ys) == 0:
        return None
    return (ys.mean(), xs.mean())


def run(name, d, gt):
    frames = sorted(gt.keys())
    feats = {fr: np.load(os.path.join(d, f'{fr}_f.npy')).astype(np.float32) for fr in frames}
    segs = {fr: np.load(os.path.join(d, f'{fr}_s.npy')).astype(np.int64) for fr in frames}
    fr_hits, ch_hits, n = 0.0, 0, 0
    for frA in frames:
        for obj in gt[frA]:
            q = pick_query_tile(segs[frA][0], obj['mask'])
            if q is None or q >= feats[frA].shape[0]:
                continue
            qv = feats[frA][q]
            qv = qv / (np.linalg.norm(qv) + 1e-8)
            frame_hit, best = [], None
            for frB in frames:
                if frB == frA:
                    continue
                matches = [o for o in gt[frB] if o['label'] == obj['label']]
                if not matches:
                    continue
                sims = feats[frB] @ qv
                t = int(np.argmax(sims))
                c = tile_center(segs[frB][0], t)
                hit = c is not None and any(b[0] <= c[1] <= b[2] and b[1] <= c[0] <= b[3]
                                            for b in [o['bbox'] for o in matches])
                frame_hit.append(hit)
                if best is None or sims[t] > best[0]:
                    best = (float(sims[t]), hit)
            if best is None:
                continue
            n += 1
            fr_hits += np.mean(frame_hit)
            ch_hits += int(best[1])
    print(f'[{SCENE}] {name:14s} frame-hit-rate {fr_hits / n:.2%}  chosen {ch_hits}/{n} = {ch_hits / n:.2%}')


if __name__ == '__main__':
    gt = load_gt()
    for name, d in DIRS.items():
        if os.path.isdir(d):
            run(name, d, gt)
        else:
            print(f'[{SCENE}] {name:14s} MISSING {d}')
