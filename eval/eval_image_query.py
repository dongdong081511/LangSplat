"""Image-query 3D localization eval (task-routing axis, EXP-035).

For each GT object in frame A: take its largest-overlap tile feature (AE-encoded,
from --query_subdir) as a visual query; retrieve on frame B's rendered feature
field; hit if the peak point falls inside frame B's same-label GT bbox.

Queries go through the same AE as the 3D field, mirroring the real pipeline.
"""
import json
import glob
import os
import argparse
import numpy as np
import cv2


def load_gt(gt_dir):
    gt = {}
    for js in sorted(glob.glob(os.path.join(gt_dir, 'frame_*.json'))):
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
    counts = [(int(t), int((seg[obj_mask > 0] == t).sum())) for t in idxs]
    counts.sort(key=lambda x: -x[1])
    return counts[0][0]


def aggregate_renders(src_dir, dst_dir, k):
    """EXP-049: pixel-wise neighbor-frame averaging of rendered feature maps.
    Adjacent video frames are near-aligned; cross-frame tile tids are NOT aligned
    (SAM per-frame), so tile-level aggregation is invalid — pixels only."""
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, '*.npy')))
    assert len(files) > 100, f'too few renders in {src_dir}'
    for i, fp in enumerate(files):
        out = os.path.join(dst_dir, os.path.basename(fp))
        if os.path.exists(out):
            continue
        acc = np.load(fp).astype(np.float32)
        cnt = 1
        for j in range(1, k + 1):
            for dj in (-j, j):
                ii = i + dj
                if 0 <= ii < len(files):
                    acc += np.load(files[ii]).astype(np.float32)
                    cnt += 1
        np.save(out, (acc / cnt).astype(np.float32))
    print(f'aggregated k={k}: {len(files)} frames -> {dst_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--scene_root', required=True)
    ap.add_argument('--db_render_dirs', nargs=3, required=True,
                    help='render level 1/2/3 dirs containing train/ours_None/renders_npy')
    ap.add_argument('--query_subdir', required=True,
                    help='AE-encoded tile features dir (same space as renders)')
    ap.add_argument('--seg_indexs', nargs=3, type=int, default=[0, 1, 2],
                    help='seg layer index for each render level')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out_json', default='', help='save per-pair results (for McNemar paired test)')
    ap.add_argument('--query_from_render', action='store_true',
                    help='EXP-053: query = mean of frame-A render features inside GT mask (same space as db), bypassing 2D tile AE codes')
    ap.add_argument('--query_tile_mask', action='store_true',
                    help='EXP-053b: with --query_from_render, use the SAM tile region (instead of full GT mask) for the render mean')
    args = ap.parse_args()

    gt = load_gt(args.gt_dir)
    frames = sorted(gt.keys())
    n_obj = sum(len(v) for v in gt.values())
    print(f'[{args.tag}] GT frames: {len(frames)}, objects: {n_obj}')

    feats, segs = {}, {}
    for fr in frames:
        feats[fr] = np.load(os.path.join(args.scene_root, args.query_subdir,
                                         f'{fr}_f.npy')).astype(np.float32)
        segs[fr] = np.load(os.path.join(args.scene_root, args.query_subdir,
                                        f'{fr}_s.npy')).astype(np.int64)

    kernel = np.ones((30, 30)) / 900
    stats = {li: [0, 0, 0, 0] for li in range(3)}  # n, hit_first, hit_any, (unused)
    chosen = [0, 0, 0]  # n, hit_first, hit_any
    multi = [0, 0, 0.0]  # EXP-048: n_mf_all_any, n_mf_majority, sum_mf_rate
    lv_any = [0, 0]  # EXP-060: level-any (n, hit_any) — 任一 level top1 命中即算
    lv_vote = [0, 0]  # EXP-060: level-vote (n, hit_any) — 过半 level top1 命中即算
    records = []  # per-pair results for McNemar

    for frA in frames:
        for i_obj, obj in enumerate(gt[frA]):
            best = None  # (sim, li, frB, pt, bboxes)
            level_best = {}  # li -> (sim, frB, pt, bboxes)
            frame_hits = []  # EXP-048: per-(li,frB) top1 hit_any, for multi-frame aggregation
            for li in range(3):
                si = args.seg_indexs[li]
                if args.query_from_render:
                    if obj['mask'].sum() < 30:
                        continue
                    rpA = os.path.join(args.db_render_dirs[li], 'train', 'ours_None',
                                       'renders_npy', f'{int(frA.split("_")[-1]) - 1:05d}.npy')
                    RA = np.load(rpA).astype(np.float32)
                    if args.query_tile_mask:
                        q = pick_query_tile(segs[frA][si], obj['mask'])
                        if q is None or q >= feats[frA].shape[0]:
                            continue
                        qm = (segs[frA][si] == q)
                        if qm.sum() < 30:
                            continue
                        qv = RA[qm].mean(axis=0)
                    else:
                        qv = RA[obj['mask'] > 0].mean(axis=0)
                else:
                    q = pick_query_tile(segs[frA][si], obj['mask'])
                    if q is None or q >= feats[frA].shape[0]:
                        continue
                    qv = feats[frA][q]
                qv = qv / (np.linalg.norm(qv) + 1e-8)
                for frB in frames:
                    if frB == frA:
                        continue
                    matches = [o for o in gt[frB] if o['label'] == obj['label']]
                    if not matches:
                        continue
                    rp = os.path.join(args.db_render_dirs[li], 'train', 'ours_None',
                                      'renders_npy', f'{int(frB.split("_")[-1]) - 1:05d}.npy')
                    R = np.load(rp).astype(np.float32)
                    Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
                    sim = Rn.reshape(-1, R.shape[-1]) @ qv
                    simf = cv2.filter2D(sim.reshape(R.shape[:2]), -1, kernel)
                    pt = np.unravel_index(np.argmax(simf), simf.shape)
                    bboxes = [o['bbox'] for o in matches]
                    cand = (float(simf[pt]), li, frB, (pt[0], pt[1]), bboxes)
                    y0, x0 = cand[3]
                    frame_hits.append(any(b[0] <= x0 <= b[2] and b[1] <= y0 <= b[3] for b in bboxes))
                    if best is None or cand[0] > best[0]:
                        best = cand
                    lb = level_best.get(li)
                    if lb is None or cand[0] > lb[0]:
                        level_best[li] = (cand[0], frB, cand[3], bboxes)
            if best is None:
                continue
            _, li, frB, pt, bboxes = best
            y, x = pt
            hit_first = bboxes[0][0] <= x <= bboxes[0][2] and bboxes[0][1] <= y <= bboxes[0][3]
            hit_any = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bboxes)
            chosen[0] += 1
            chosen[1] += int(hit_first)
            chosen[2] += int(hit_any)
            mf_all = int(any(frame_hits))          # EXP-048: any of per-frame top1 hits
            mf_maj = int(np.mean(frame_hits) >= 0.5)  # EXP-048: majority of per-frame top1 hits
            mf_rate = float(np.mean(frame_hits))      # EXP-048: per-frame hit rate
            multi[0] += mf_all
            multi[1] += mf_maj
            multi[2] += mf_rate
            records.append({'frameA': frA, 'obj_i': i_obj, 'label': obj['label'],
                            'frameB': frB, 'level': li + 1, 'sim': best[0],
                            'hit_first': int(hit_first), 'hit_any': int(hit_any),
                            'mf_all_any': mf_all, 'mf_majority': mf_maj,
                            'mf_rate': round(mf_rate, 4)})
            for l2, (sim2, frB2, pt2, bboxes2) in level_best.items():
                y2, x2 = pt2
                hf = bboxes2[0][0] <= x2 <= bboxes2[0][2] and bboxes2[0][1] <= y2 <= bboxes2[0][3]
                ha = any(b[0] <= x2 <= b[2] and b[1] <= y2 <= b[3] for b in bboxes2)
                stats[l2][0] += 1
                stats[l2][1] += int(hf)
                stats[l2][2] += int(ha)
            # EXP-060: cross-level aggregation (fair comparison requires running both fields)
            if level_best:
                lv_hits = []
                for l2, (sim2, frB2, pt2, bboxes2) in level_best.items():
                    y2, x2 = pt2
                    ha2 = any(b[0] <= x2 <= b[2] and b[1] <= y2 <= b[3] for b in bboxes2)
                    lv_hits.append(int(ha2))
                lv_any[0] += 1
                lv_any[1] += int(any(lv_hits))
                lv_vote[0] += 1
                lv_vote[1] += int(np.mean(lv_hits) >= 0.5)

    print(f'[{args.tag}] query_subdir={args.query_subdir}')
    print(f'[{args.tag}] chosen-level top1: {chosen[1]}/{chosen[0]} = '
          f'{chosen[1] / max(chosen[0], 1):.2%}')
    print(f'[{args.tag}] chosen-level top1 (any-bbox): {chosen[2]}/{chosen[0]} = '
          f'{chosen[2] / max(chosen[0], 1):.2%}')
    n = max(chosen[0], 1)
    print(f'[{args.tag}] EXP-048 multi-frame: all-any {multi[0]}/{n} = {multi[0] / n:.2%}  '
          f'majority {multi[1]}/{n} = {multi[1] / n:.2%}  hit-rate {multi[2] / n:.2%}')
    print(f'[{args.tag}] EXP-060 level-any: {lv_any[1]}/{lv_any[0]} = '
          f'{lv_any[1] / max(lv_any[0], 1):.2%}  level-vote: {lv_vote[1]}/{lv_vote[0]} = '
          f'{lv_vote[1] / max(lv_vote[0], 1):.2%}')
    for li in range(3):
        n, hf, ha, _ = stats[li]
        print(f'[{args.tag}] level{li + 1} pairs: {n} top1(first)={hf / max(n, 1):.2%} '
              f'top1(any)={ha / max(n, 1):.2%}')
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump({'tag': args.tag, 'query_subdir': args.query_subdir,
                       'n_pairs': chosen[0], 'pairs': records}, f)
        print(f'[{args.tag}] saved {len(records)} pairs -> {args.out_json}')


if __name__ == '__main__':
    main()
