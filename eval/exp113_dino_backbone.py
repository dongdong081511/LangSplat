# EXP-113 (P2-1): DINOv2 backbone-scale transfer, ViT-B/14 (768d) -> ViT-L/14 (1024d).
# 113a: 2D-teacher endpoint under the EXP-082 exhaustive frame-pair probe, on L14 tile
#       features (language_features_dino_l14, same SAM tiles verified via _s.npy).
# 113b: 3D raw-field endpoint on the trained L14 32d field (when renders exist).
# Protocol verbatim from exp082 (KERNEL 30x30, seg_query, render_query).
import os, sys, json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exp082_attribution import (gt_frames, seg_query, render_query, render_idx,
                                KERNEL, OUT, DS)

L14_SUB = 'language_features_dino_l14'
# B/14 archived numbers (exp082/exp082b logs) for direct comparison
B14 = {'teatime': dict(hit='146/227', acc=64.3, cos=0.740),
       'figurines': dict(hit='72/112', acc=64.3, cos=0.739),
       'ramen': dict(hit='115/329', acc=35.0, cos=0.726),
       'waldo_kitchen': dict(hit='10/22', acc=45.5, cos=0.689)}
SCENES = sys.argv[1].split(',') if len(sys.argv) > 1 else ['teatime', 'waldo_kitchen']
RAW_FIELD = {s: f'{s}_dino_l14_1' for s in SCENES}


def seg_data_l14(scene, fr):
    sub = os.path.join(DS, scene, L14_SUB)
    F = np.load(os.path.join(sub, fr + '_f.npy')).astype(np.float32)
    S = np.load(os.path.join(sub, fr + '_s.npy')).astype(np.int64)[-1]
    return F, S


def eval_2d(scene):
    from collections import defaultdict
    gts = gt_frames(scene)
    fdir = os.path.join(DS, scene, L14_SUB)
    gts = [(fr, objs) for fr, objs in gts
           if os.path.exists(os.path.join(fdir, fr + '_f.npy'))]
    by_frame, views = defaultdict(list), {}
    for fr, objs in gts:
        F, S = seg_data_l14(scene, fr)
        for i, obj in enumerate(objs):
            q = seg_query(F, S, obj['mask'])
            if q is None:
                continue
            by_frame[fr].append((i, obj['label'], q))
            views.setdefault(obj['label'], {})[fr] = q
    gt_by_frame = dict(gts)
    hit = n = 0
    per_obj = defaultdict(lambda: [0, 0])
    for frB in sorted(by_frame):
        F, S = seg_data_l14(scene, frB)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        valid = S >= 0
        base = np.where(valid, np.maximum(S, 0), 0)
        gtB = gt_by_frame[frB]
        for frA, rows in by_frame.items():
            if frA == frB:
                continue
            for i, lab, q in rows:
                bbs = [o['bbox'] for o in gtB if o['label'] == lab]
                if not bbs:
                    continue
                sc = Fn @ q
                smap = sc[base].reshape(S.shape)
                smap[~valid] = -1.0
                simf = cv2.filter2D(smap, -1, KERNEL)
                y, x = np.unravel_index(np.argmax(simf), simf.shape)
                h = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bbs)
                n += 1
                hit += h
                per_obj[(frA, i, lab)][0] += h
                per_obj[(frA, i, lab)][1] += 1
    # cross-view 2D-2D cos per label (same aggregation as exp082)
    cos2d = {}
    for lab, vs in views.items():
        frs = sorted(vs)
        cs = [float(vs[a] @ vs[b])
              for ai, a in enumerate(frs) for b in frs[ai + 1:]]
        if cs:
            cos2d[lab] = float(np.mean(cs))
    return hit, n, per_obj, cos2d


def eval_3d(scene):
    from collections import defaultdict
    rdir = os.path.join(OUT, RAW_FIELD[scene], 'train', 'ours_None', 'renders_npy')
    if not os.path.isdir(rdir):
        return None
    gts = gt_frames(scene)
    has3d = [(fr, objs) for fr, objs in gts
             if os.path.exists(os.path.join(rdir, f'{render_idx(fr):05d}.npy'))]
    by_frame, views = defaultdict(list), {}
    for fr, objs in has3d:
        R = np.load(os.path.join(rdir, f'{render_idx(fr):05d}.npy')).astype(np.float32)
        for i, obj in enumerate(objs):
            q = render_query(R, obj['mask'])
            by_frame[fr].append((i, obj['label'], q))
            views.setdefault(obj['label'], {})[fr] = q
    gt_by_frame = dict(gts)
    hit = n = 0
    for frB in sorted(by_frame):
        R = np.load(os.path.join(rdir, f'{render_idx(frB):05d}.npy')).astype(np.float32)
        Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
        gtB = gt_by_frame[frB]
        for frA, rows in by_frame.items():
            if frA == frB:
                continue
            for i, lab, q in rows:
                bbs = [o['bbox'] for o in gtB if o['label'] == lab]
                if not bbs:
                    continue
                smap = (Rn @ q).reshape(R.shape[:2])
                simf = cv2.filter2D(smap, -1, KERNEL)
                y, x = np.unravel_index(np.argmax(simf), simf.shape)
                h = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bbs)
                n += 1
                hit += h
    # cross-view 3D cos
    cos3d = {}
    for lab, vs in views.items():
        frs = sorted(vs)
        cs = [float(vs[a] @ vs[b])
              for ai, a in enumerate(frs) for b in frs[ai + 1:]]
        if cs:
            cos3d[lab] = float(np.mean(cs))
    return hit, n, cos3d


def main():
    report = {}
    for scene in SCENES:
        h2, n2, per_obj, cos2d = eval_2d(scene)
        acc2 = 100 * h2 / max(n2, 1)
        mean_cos = float(np.mean(list(cos2d.values()))) if cos2d else None
        b = B14.get(scene, {})
        print(f"\n===== [{scene}] 2D teacher L14: hit={h2}/{n2} ({acc2:.1f}%) "
              f"cos2d={mean_cos if mean_cos is None else round(mean_cos, 4)} | "
              f"B/14 archived: {b.get('hit')} ({b.get('acc')}%) cos={b.get('cos')}")
        per = defaultdict(lambda: [0, 0])
        for (frA, i, lab), (h, nn) in per_obj.items():
            per[lab][0] += h
            per[lab][1] += nn
        for lab, (h, nn) in sorted(per.items()):
            print(f"    {lab:26s} L14 {h}/{nn}  ({100*h/nn:.0f}%)")
        r = dict(l14=dict(hit=f'{h2}/{n2}', acc=acc2, cos2d=mean_cos,
                          per_object={k: v for k, v in per.items()}),
                 b14=b)
        r3 = eval_3d(scene)
        if r3:
            h3, n3, cos3d = r3
            acc3 = 100 * h3 / max(n3, 1)
            mc3 = float(np.mean(list(cos3d.values()))) if cos3d else None
            print(f"[{scene}] 3D raw field L14: hit={h3}/{n3} ({acc3:.1f}%) "
                  f"cos3d={mc3 if mc3 is None else round(mc3, 4)} | lift {acc3-acc2:+.1f}pp")
            r['l14_3d_raw'] = dict(hit=f'{h3}/{n3}', acc=acc3, cos3d=mc3)
        report[scene] = r
    out = os.path.join(BASE := os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..')),
        'eval_result', 'exp113_dino_backbone.json')
    json.dump(report, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
