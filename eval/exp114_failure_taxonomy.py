# EXP-114 (P2-3): failure-case taxonomy and scene-attribute analysis.
# Joins four evidence sources per object / per prompt:
#   1. GT label jsons  -> object size (mask area fraction), same-label instance count
#   2. EXP-082/082b log -> per-object 2D-teacher acc, 3D-field acc, cross-view cos
#   3. EXP-110 json     -> per-object cross-view offset repair rate + mean 2D offset
#   4. EXP-106 records  -> per-prompt text-side localization (far-jump errors, IoU)
# Output: taxonomy classes + Spearman correlations -> eval_result/exp114_failure_taxonomy.json
import os, sys, glob, json, re
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from exp082_attribution import gt_frames, DS

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
LOGS = {'teatime': 'exp082_attribution.log',
        'figurines': 'exp082b_figurines_attr.log',
        'ramen': 'exp082_attribution.log',
        'waldo_kitchen': 'exp082_attribution.log'}
RECORDS = {'teatime': 'records_teatime_8d.json',
           'figurines': 'records_figurines_24d.json',
           'ramen': 'records_ramen_clip8d.json',
           'waldo_kitchen': 'records_waldo_kitchen_clip8d.json'}
FAR_PX = 150          # loc_err above this = discriminative far jump
TIE_COS = 0.85        # 2D cross-view cos above this = low-variance (tie) object
SMALL_FRAC = 0.005    # mask area < 0.5% of frame = small object

OBJ_RE = re.compile(r'^\s{4}(.+?)\s{2,}(\d+)/(\d+)\s+(\d+)/(\d+)\s*$')
COS_RE = re.compile(r'^\s{4}(.+?)\s+views=\s*(\d+)\s+2D ([\d.]+)\s+3D ([\d.]+)')


def parse_attr_log(path):
    """Parse exp082-style log: per-object hit counts + per-object cross-view cos."""
    hit, cos = {}, {}
    scene = None
    for line in open(path):
        m = re.match(r'===== \[(\S+)\] feat=', line)
        if m:
            scene = m.group(1)
            continue
        m = OBJ_RE.match(line.rstrip('\n'))
        if m and scene:
            lab = m.group(1).strip()
            hit.setdefault(scene, {})[lab] = dict(
                d2=int(m.group(2)), n2=int(m.group(3)),
                d3=int(m.group(4)), n3=int(m.group(5)))
            continue
        m = COS_RE.match(line.rstrip('\n'))
        if m and scene:
            hit.setdefault(scene, {}).setdefault(m.group(1).strip(), {}) \
               .update(cos2d=float(m.group(3)), cos3d=float(m.group(4)))
    return hit


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else None


def exp110_per_object():
    d = json.load(open(os.path.join(BASE, 'eval_result', 'exp110_consistency_dist.json')))
    out = {}
    for sc, rep in d.items():
        if sc == 'pooled':
            continue
        per = {}
        for p in rep['pairs']:
            o = per.setdefault(p['label'], dict(n=0, rep=0, off2d=[]))
            o['n'] += 1
            o['rep'] += int(p['offsm'] < p['off2d'])
            o['off2d'].append(p['off2d'])
        out[sc] = {lab: dict(repair=o['rep'] / o['n'], off2d=float(np.mean(o['off2d'])), n=o['n'])
                   for lab, o in per.items()}
    return out


def gt_attributes(scene):
    """Per-label: median mask area fraction, max same-label instance count, n_frames."""
    per = {}
    for fr, objs in gt_frames(scene):
        h, w = objs[0]['mask'].shape
        counts = {}
        for o in objs:
            counts[o['label']] = counts.get(o['label'], 0) + 1
        for o in objs:
            lab = o['label']
            frac = float(o['mask'].sum()) / (h * w)
            a = per.setdefault(lab, dict(fracs=[], ninst=0, nframes=0))
            a['fracs'].append(frac)
            a['nframes'] += 1
            a['ninst'] = max(a['ninst'], counts[lab])
    return {lab: dict(area_frac=float(np.median(a['fracs'])), n_inst=a['ninst'],
                      n_frames=a['nframes'])
            for lab, a in per.items()}


def prompt_failures(scene):
    """Per-prompt text-side failure stats from EXP-106 records (ours deployed system)."""
    f = os.path.join(BASE, 'eval_result', 'pareto_locerr', RECORDS[scene])
    d = json.load(open(f))
    per = {}
    for r in d['records']:
        o = per.setdefault(r['prompt'], dict(n=0, hit=0, far=0, iou=[]))
        o['n'] += 1
        o['hit'] += int(r['loc_hit'])
        o['far'] += int(r['loc_err_px'] > FAR_PX)
        o['iou'].append(r['chosen_iou'])
    return {p: dict(n=o['n'], loc_acc=o['hit'] / o['n'], far_rate=o['far'] / o['n'],
                    iou_mean=float(np.mean(o['iou'])))
            for p, o in per.items()}


def classify(obj):
    """Assign failure-mechanism class(es) from quantitative attributes."""
    tags = []
    if obj.get('cos2d') is not None and obj['cos2d'] >= TIE_COS:
        tags.append('tie(low-variance)')
    if obj['area_frac'] < SMALL_FRAC:
        tags.append('small')
    if obj['n_inst'] >= 2:
        tags.append('multi-instance')
    if obj.get('d2') is not None and obj['n2'] >= 5:
        r2, r3 = obj['d2'] / obj['n2'], (obj.get('d3') or 0) / max(obj.get('n3') or 1, 1)
        if r2 < 0.6 and r3 >= 0.9:
            tags.append('aggregation-repairable')
        elif r2 < 1.0 and r3 - r2 <= 0.1:
            tags.append('entry-negative')  # 3D aggregation fails to lift (n>=5, not saturated)
    return tags


def main():
    attr = parse_attr_log(os.path.join(BASE, 'eval_result', 'exp082_attribution.log'))
    attr.update(parse_attr_log(os.path.join(BASE, 'eval_result', 'exp082b_figurines_attr.log')))
    r110 = exp110_per_object()
    report = {}
    X = {k: [] for k in ('area', 'cos2d', 'off2d', 'r2', 'r3', 'ninst')}
    Y = {k: [] for k in ('repair', 'lift')}
    for scene in ['teatime', 'figurines', 'ramen', 'waldo_kitchen']:
        gtatt = gt_attributes(scene)
        pf = prompt_failures(scene)
        objs = []
        for lab, a in sorted(gtatt.items()):
            hit = attr.get(scene, {}).get(lab, {})
            r = r110.get(scene, {}).get(lab, {})
            obj = dict(label=lab, area_frac=round(a['area_frac'], 5),
                       n_inst=a['n_inst'], n_frames=a['n_frames'],
                       d2=hit.get('d2'), n2=hit.get('n2'),
                       d3=hit.get('d3'), n3=hit.get('n3'),
                       cos2d=hit.get('cos2d'), cos3d=hit.get('cos3d'),
                       repair110=r.get('repair'), off2d110=r.get('off2d'),
                       prompt=pf.get(lab))
            obj['class'] = classify(obj)
            if obj['d2'] and obj['n2']:
                X['area'].append(a['area_frac']); X['cos2d'].append(obj['cos2d'])
                X['off2d'].append(r.get('off2d', np.nan)); X['ninst'].append(a['n_inst'])
                r2 = obj['d2'] / obj['n2']
                r3 = obj['d3'] / obj['n3'] if obj.get('d3') else np.nan
                X['r2'].append(r2); X['r3'].append(r3)
                Y['lift'].append(r3 - r2)
                if not np.isnan(r.get('repair', np.nan)):
                    Y['repair'].append(r['repair'])
                    # keep aligned pair for cos2d-vs-repair
            objs.append(obj)
        report[scene] = dict(n_objects=len(objs), objects=objs,
                             prompt_failures=pf)
        print(f"\n===== [{scene}] {len(objs)} objects =====")
        for o in objs:
            r2 = f"{o['d2']}/{o['n2']}" if o['d2'] is not None else '--'
            r3 = f"{o['d3']}/{o['n3']}" if o['d3'] is not None else '--'
            rep = f"{o['repair110']:.2f}" if o.get('repair110') is not None else '--'
            pf = o.get('prompt')
            far = f"far={pf['far_rate']:.2f}" if pf else ''
            print(f"  {o['label']:26s} area={o['area_frac']*100:5.2f}% inst={o['n_inst']} "
                  f"2D={r2:8s} 3D={r3:8s} cos2d={o['cos2d'] or 0:.3f} rep110={rep} "
                  f"{far} [{'|'.join(o['class'])}]")
    # correlations
    corr = {}
    pairs = [('area_frac vs 2D acc', X['area'], X['r2']),
             ('cos2d vs lift', X['cos2d'], Y['lift']),
             ('off2d(110) vs 2D acc', X['off2d'], X['r2']),
             ('off2d(110) vs cos2d', X['off2d'], X['cos2d']),
             ('n_inst vs 2D acc', X['ninst'], X['r2'])]
    for name, a, b in pairs:
        m = ~(np.isnan(np.asarray(a, float)) | np.isnan(np.asarray(b, float)))
        rho = spearman(np.asarray(a)[m], np.asarray(b)[m]) if m.sum() >= 4 else None
        corr[name] = dict(rho=rho, n=int(m.sum()))
        print(f"corr {name}: rho={rho if rho is None else round(rho, 3)} (n={m.sum()})")
    report['correlations'] = corr
    out = os.path.join(BASE, 'eval_result', 'exp114_failure_taxonomy.json')
    json.dump(report, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
