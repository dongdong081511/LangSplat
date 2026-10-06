"""Offline gate-threshold study for EXP-043 (spatial-adaptive smoothing).

Loads per-tile signals dumped by smooth_tiles_adaptive.py (--dry) and reports:
  1. per-scene/level signal distributions (quantiles) for cos2d and cons
  2. for candidate hard gates, the gated-out fraction per scene/level
Desired behavior: waldo (bad renders, level1 collapse) should gate OUT a large
fraction especially at level1; healthy scenes (teatime/figurines/ramen) should
keep most tiles (behavior ~ global alpha=0.7). Level index li=0 -> render level1
(coarse), li=2 -> level3 (fine).
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = ['teatime', 'figurines', 'waldo_kitchen', 'ramen']
TAUS = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
QS = [0.2, 0.3, 0.5]

data = {}
for sc in SCENES:
    p = os.path.join(ROOT, 'eval_result', 'adaptive', f'signals_{sc}.json')
    if os.path.exists(p):
        data[sc] = json.load(open(p))
    else:
        print(f'missing {p}, skip')

for sc, d in data.items():
    t = d['tiles']
    print(f'\n== {sc}: {len(t["cos2d"])} tiles')
    for li in range(3):
        sel = [i for i, l in enumerate(t['li']) if l == li]
        if not sel:
            continue
        for sig in ['cos2d', 'cons']:
            v = np.array([t[sig][i] for i in sel])
            print(f'  L{li + 1} {sig}: n={len(sel)} q05/25/50/75/95 = '
                  f'{[round(float(x), 3) for x in np.quantile(v, [.05, .25, .5, .75, .95])]}')

print('\n== hard gate: gated-out fraction L1/L2/L3 (scene abbr)')
for sig in ['cos2d', 'cons']:
    print(f'-- signal={sig}')
    for tau in TAUS:
        row = []
        for sc in SCENES:
            if sc not in data:
                continue
            t = data[sc]['tiles']
            v = np.array(t[sig])
            l = np.array(t['li'])
            fr = [(v[l == k] < tau).mean() for k in range(3)]
            row.append(f'{sc[:4]} {fr[0]:.2f}/{fr[1]:.2f}/{fr[2]:.2f}')
        print(f'  tau={tau}: ' + ' | '.join(row))

print('\n== quantile gate (scene-level q): gated-out tile count composition L1/L2/L3')
for sig in ['cos2d', 'cons']:
    for q in QS:
        row = []
        for sc in SCENES:
            if sc not in data:
                continue
            t = data[sc]['tiles']
            v = np.array(t[sig])
            l = np.array(t['li'])
            tau = np.quantile(v, q)
            comp = [int((l[v < tau] == k).sum()) for k in range(3)]
            row.append(f'{sc[:4]} {comp[0]}/{comp[1]}/{comp[2]} of {len(v)}')
        print(f'-- {sig} q={q}: ' + ' | '.join(row))
