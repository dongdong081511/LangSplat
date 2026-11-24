# EXP-110 (P2-2): cross-view feature-offset DISTRIBUTIONS (multi-object, multi-view).
# Extends EXP-082's scene-mean consistency to per-object x per-view-pair offsets:
#   off_2d  = 1 - cos(q2d_i, q2d_j)      (2D teacher tile features)
#   off_raw = 1 - cos(qraw_i, qraw_j)    (raw 3D field renders, dino_32d)
#   off_sm  = 1 - cos(qsm_i, qsm_j)      (self-distilled field, dino_32ds_a07)
# repair rates: fraction of pairs where the offset shrinks vs the 2D teacher
# (3D aggregation) and vs the raw field (self-distillation increment).
# Reuses EXP-082's query definitions verbatim (seg_query / render_query / level-1).
import os, sys, glob, json
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exp082_attribution import gt_frames, seg_data, seg_query, render_query, render_idx

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DS = os.path.join(BASE, 'dataset', 'lerf_ovs')
OUT = os.path.join(BASE, 'output')

SCENES = ['teatime', 'figurines', 'ramen', 'waldo_kitchen']
RAW_F, SM_F = 'dino_32d', 'dino_32ds_a07'


def collect(scene):
    gts = gt_frames(scene)
    fdir = os.path.join(DS, scene, 'language_features_dino')
    gts = [(fr, objs) for fr, objs in gts
           if os.path.exists(os.path.join(fdir, fr + '_f.npy'))]
    rdr = {k: os.path.join(OUT, f'{scene}_{k}_1', 'train', 'ours_None', 'renders_npy')
           for k in (RAW_F, SM_F)}
    views = {}  # label -> {fr: (q2d, qraw, qsm)}
    cache = {}
    for fr, objs in gts:
        ri = render_idx(fr)
        paths = {k: os.path.join(rdr[k], f'{ri:05d}.npy') for k in rdr}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        F, S = seg_data(scene, 'dino', fr)
        Rs = {k: np.load(paths[k]).astype(np.float32) for k in rdr}
        for i, obj in enumerate(objs):
            m = obj['mask']
            q2 = seg_query(F, S, m)
            if q2 is None:
                continue
            qr = render_query(Rs[RAW_F], m)
            qs = render_query(Rs[SM_F], m)
            views.setdefault(obj['label'], {})[fr] = (q2, qr, qs)
    return views


def main():
    report = {}
    pool = {'off2d': [], 'offraw': [], 'offsm': []}
    for scene in SCENES:
        views = collect(scene)
        pairs = []
        for lab, vs in sorted(views.items()):
            frs = sorted(vs)
            for a in range(len(frs)):
                for b in range(a + 1, len(frs)):
                    q2a, qra, qsa = vs[frs[a]]
                    q2b, qrb, qsb = vs[frs[b]]
                    off2d = 1 - float(q2a @ q2b)
                    offraw = 1 - float(qra @ qrb)
                    offsm = 1 - float(qsa @ qsb)
                    pairs.append(dict(label=lab, fa=frs[a], fb=frs[b],
                                      off2d=off2d, offraw=offraw, offsm=offsm))
        if not pairs:
            print(f'[{scene}] no usable pairs'); continue
        o2 = np.array([p['off2d'] for p in pairs])
        oraw = np.array([p['offraw'] for p in pairs])
        osm = np.array([p['offsm'] for p in pairs])
        rep_raw = float((oraw < o2).mean())
        rep_sm = float((osm < o2).mean())
        rep_sm_raw = float((osm < oraw).mean())
        n_obj = len(views)
        q = lambda x: [round(float(v), 4) for v in np.percentile(x, [25, 50, 75])]
        report[scene] = dict(
            n_objects=n_obj, n_pairs=len(pairs),
            off2d=dict(mean=float(o2.mean()), iqr=q(o2)),
            offraw=dict(mean=float(oraw.mean()), iqr=q(oraw)),
            offsm=dict(mean=float(osm.mean()), iqr=q(osm)),
            repair_raw_3daggregation=rep_raw,
            repair_smooth_vs_2d=rep_sm,
            repair_smooth_vs_raw=rep_sm_raw,
            pairs=pairs)
        pool['off2d'] += o2.tolist(); pool['offraw'] += oraw.tolist(); pool['offsm'] += osm.tolist()
        print(f"[{scene}] objects={n_obj} pairs={len(pairs)} | "
              f"off 2D {o2.mean():.4f} -> raw {oraw.mean():.4f} -> smooth {osm.mean():.4f} | "
              f"repair(3D agg) {100*rep_raw:.1f}% | repair(smooth vs 2D) {100*rep_sm:.1f}% | "
              f"repair(smooth vs raw) {100*rep_sm_raw:.1f}%")
        # per-object repair table
        per = {}
        for p in pairs:
            d = per.setdefault(p['label'], [0, 0])
            d[0] += int(p['offsm'] < p['off2d']); d[1] += 1
        for lab, (r, n) in sorted(per.items()):
            print(f"    {lab:28s} pairs={n:3d} repaired {r}/{n} ({100*r/n:.0f}%)")

    if pool['off2d']:
        o2 = np.array(pool['off2d']); oraw = np.array(pool['offraw']); osm = np.array(pool['offsm'])
        print(f"\nPOOLED ({len(o2)} pairs): off 2D {o2.mean():.4f} -> raw {oraw.mean():.4f} "
              f"-> smooth {osm.mean():.4f} | repair(3D agg) {100*(oraw<o2).mean():.1f}% | "
              f"repair(smooth vs 2D) {100*(osm<o2).mean():.1f}% | "
              f"repair(smooth vs raw) {100*(osm<oraw).mean():.1f}%")
        report['pooled'] = dict(n_pairs=len(o2),
                                off2d=float(o2.mean()), offraw=float(oraw.mean()), offsm=float(osm.mean()),
                                repair_raw_3daggregation=float((oraw < o2).mean()),
                                repair_smooth_vs_2d=float((osm < o2).mean()),
                                repair_smooth_vs_raw=float((osm < oraw).mean()))
    out = os.path.join(BASE, 'eval_result', 'exp110_consistency_dist.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
