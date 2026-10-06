"""Spatial-adaptive self-distillation smoothing (EXP-043).

smooth_tiles.py blends render (3D-consistent) features into 2D tile supervision
with a GLOBAL alpha, which hurts scenes whose DINO-field renders are unreliable
(waldo: alpha=0.7 -> -7.7pp, level1 collapse). Here every (frame, level, tile)
gets its own alpha from a per-tile confidence signal:

  cos2d : cos(mean render feature of tile, 2D supervision feature)
          (render-2D agreement at this tile)
  cons  : mean over other SAM levels l' of cos(rm_li, mean render of l' on the
          same pixel mask)  (cross-level field consensus = render trustworthiness;
          should flag waldo-style single-level collapse while sparing healthy
          view-sensitive tiles that smoothing is meant to fix)

Gate modes (--gate):
  none     constant alpha (reproduces smooth_tiles.py behavior)
  hard     alpha if signal >= tau else beta*alpha          (tau = --tau)
  soft     alpha * clamp((signal-tau)/(tau1-tau), 0, 1)    (ramp --tau..--tau1)
  quantile hard rule with tau = scene-level --quantile quantile of the signal

--dry computes signals + stats only (no features written) for the threshold
study. The stats JSON always records BOTH signals per tile regardless of the
chosen --signal, so gate variants can be compared offline without re-running.
"""
import os
import glob
import json
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scene', required=True)
ap.add_argument('--alpha', type=float, default=0.7, help='max weight of render features')
ap.add_argument('--beta', type=float, default=0.0, help='alpha scale for gated-out tiles')
ap.add_argument('--signal', default='cons', choices=['none', 'cos2d', 'cons'])
ap.add_argument('--gate', default='hard', choices=['none', 'hard', 'soft', 'quantile'])
ap.add_argument('--tau', type=float, default=0.8, help='hard threshold / soft ramp lower bound')
ap.add_argument('--tau1', type=float, default=0.95, help='soft ramp upper bound')
ap.add_argument('--quantile', type=float, default=0.3, help='quantile mode: tau at this scene-level q')
ap.add_argument('--min_area', type=int, default=30)
ap.add_argument('--src_subdir', default='language_features_dim32_dino')
ap.add_argument('--render_base', default=None, help='default {scene}_dino_32d')
ap.add_argument('--iter_tag', required=True, help='output subdir suffix (skipped when --dry)')
ap.add_argument('--stats_out', default='', help='per-tile signals JSON (both signals recorded)')
ap.add_argument('--dry', action='store_true', help='signals + stats only, write no features')
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = os.path.join(ROOT, 'dataset', 'lerf_ovs', args.scene)
SRC = os.path.join(SR, args.src_subdir)
DST = os.path.join(SR, f'{args.src_subdir}_smooth_{args.iter_tag}')
render_base = args.render_base or f'{args.scene}_dino_32d'


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


frames = sorted(glob.glob(SRC + '/*_f.npy'))
assert frames, f'no features under {SRC}'

# ---- pass 1: per-tile signals (store unit render means for pass 2)
rec = {'fr': [], 'li': [], 't': [], 'area': [], 'cos2d': [], 'cons': []}
rms = {}  # (frame, li, t) -> unit render-mean vector
for fp in frames:
    b = os.path.basename(fp).replace('_f.npy', '')
    idx = int(b.split('_')[-1]) - 1
    f = np.load(fp).astype(np.float32)
    seg = np.load(os.path.join(SRC, f'{b}_s.npy')).astype(np.int64)
    R = []
    for li in range(3):
        rp = os.path.join(ROOT, 'output', f'{render_base}_{li + 1}', 'train',
                          'ours_None', 'renders_npy', f'{idx:05d}.npy')
        R.append(norm(np.load(rp).astype(np.float32)) if os.path.exists(rp) else None)
    for li in range(3):
        if R[li] is None:
            continue
        s = seg[li]
        for t in np.unique(s):
            if t < 0 or t >= f.shape[0]:
                continue
            m = (s == t)
            area = int(m.sum())
            if area < args.min_area:
                continue
            rm = unit(R[li][m].mean(axis=0))
            cross = [float(np.dot(rm, unit(R[l2][m].mean(axis=0))))
                     for l2 in range(3) if l2 != li and R[l2] is not None]
            cos2d = float(np.dot(rm, unit(f[t])))
            cons = float(np.mean(cross)) if cross else cos2d
            rec['fr'].append(b)
            rec['li'].append(li)
            rec['t'].append(int(t))
            rec['area'].append(area)
            rec['cos2d'].append(cos2d)
            rec['cons'].append(cons)
            rms[(b, li, int(t))] = rm.astype(np.float32)

sig = np.array(rec['cos2d' if args.signal == 'cos2d' else 'cons'], np.float64)
if args.signal == 'none':
    tau_eff = None
elif args.gate == 'quantile':
    tau_eff = float(np.quantile(sig, args.quantile))
else:
    tau_eff = args.tau


def gate_alpha(s):
    if args.signal == 'none' or args.gate == 'none':
        return args.alpha
    if args.gate in ('hard', 'quantile'):
        return args.alpha if s >= tau_eff else args.beta * args.alpha
    g = min(max((s - tau_eff) / max(args.tau1 - args.tau, 1e-6), 0.0), 1.0)
    return args.alpha * g


alphas = [gate_alpha(s) for s in sig]

# ---- pass 2: write blended features
n_changed = 0
if not args.dry:
    os.makedirs(DST, exist_ok=True)
    by_frame = {}
    for b, li, t, a in zip(rec['fr'], rec['li'], rec['t'], alphas):
        by_frame.setdefault(b, []).append((li, t, a))
    for fp in frames:
        b = os.path.basename(fp).replace('_f.npy', '')
        f = np.load(fp).astype(np.float32)
        seg = np.load(os.path.join(SRC, f'{b}_s.npy'))
        newf = f.copy()
        for li, t, a in by_frame.get(b, []):
            newf[t] = a * rms[(b, li, t)] + (1 - a) * f[t]
        newf = norm(newf)
        np.save(os.path.join(DST, f'{b}_f.npy'), newf.astype(np.float32))
        np.save(os.path.join(DST, f'{b}_s.npy'), seg.astype(np.float32))
        n_changed += 1
    assert n_changed > 100, f'only {n_changed} frames written'

# ---- stats
per_level = {}
for li in range(3):
    sel = [i for i, l in enumerate(rec['li']) if l == li]
    if not sel:
        continue
    c2 = np.array([rec['cos2d'][i] for i in sel])
    cc = np.array([rec['cons'][i] for i in sel])
    al = np.array([alphas[i] for i in sel])
    qs = lambda v: [round(float(x), 4) for x in np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])]
    per_level[str(li)] = {
        'n': len(sel),
        'cos2d_q05_25_50_75_95': qs(c2),
        'cons_q05_25_50_75_95': qs(cc),
        'mean_alpha': round(float(al.mean()), 4),
        'frac_gated_out': round(float((al < args.alpha - 1e-9).mean()), 4),
    }
stats = {'scene': args.scene, 'args': vars(args), 'tau_effective': tau_eff,
         'n_tiles': len(rec['cos2d']), 'per_level': per_level, 'tiles': rec,
         'alphas': alphas}
if not args.dry:
    stats['frames_written'] = n_changed
if args.stats_out:
    out_dir = os.path.dirname(args.stats_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.stats_out, 'w') as fj:
        json.dump(stats, fj)
print(json.dumps(per_level, indent=1))
print(f'scene={args.scene} tau_eff={tau_eff} tiles={len(rec["cos2d"])} '
      f'dry={args.dry} frames_written={n_changed}'
      + (f' stats -> {args.stats_out}' if args.stats_out else ''))
