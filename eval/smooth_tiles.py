"""Self-distillation smoothing: blend 2D tile supervision with render-field features
(cross-view consistent) to fix DINO view sensitivity. Output new tile feature dir."""
import os, sys, glob, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scene', required=True)
ap.add_argument('--alpha', type=float, default=0.5, help='weight of render (3D-consistent) features')
ap.add_argument('--src_subdir', default='language_features_dim32_dino')
ap.add_argument('--render_base', default=None, help='render dir base under output/, default {scene}_dino_32d')
ap.add_argument('--iter_tag', required=True, help='e.g. a05; output subdir suffix')
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = os.path.join(ROOT, 'dataset', 'lerf_ovs', args.scene)
SRC = os.path.join(SR, args.src_subdir)
DST = os.path.join(SR, f'{args.src_subdir}_smooth_{args.iter_tag}')
os.makedirs(DST, exist_ok=True)
render_base = args.render_base or f'{args.scene}_dino_32d'

def norm(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

frames = sorted(glob.glob(SRC + '/*_f.npy'))
n_changed = 0
for fp in frames:
    b = os.path.basename(fp).replace('_f.npy', '')
    idx = int(b.split('_')[-1]) - 1
    f = np.load(fp).astype(np.float32)                    # [T,32] current supervision
    seg = np.load(os.path.join(SRC, f'{b}_s.npy')).astype(np.int64)  # [4,H,W]
    newf = f.copy()
    # 3 levels: render level1/2/3 <-> seg index 0/1/2 (as in eval_image_query default)
    for li in range(3):
        rp = os.path.join(ROOT, 'output', f'{render_base}_{li+1}',
                          'train', 'ours_None', 'renders_npy', f'{idx:05d}.npy')
        if not os.path.exists(rp):
            continue
        R = norm(np.load(rp).astype(np.float32))          # [H,W,32]
        s = seg[li]
        for t in np.unique(s):
            if t < 0 or t >= f.shape[0]:
                continue
            m = (s == t)
            if m.sum() < 30:
                continue
            rm = norm(R[m].mean(axis=0))
            # blend in 32d space; tile t may be touched by multiple levels -> last write wins is fine
            newf[t] = args.alpha * rm + (1 - args.alpha) * f[t]
    newf = norm(newf)
    np.save(os.path.join(DST, f'{b}_f.npy'), newf.astype(np.float32))
    np.save(os.path.join(DST, f'{b}_s.npy'), seg.astype(np.float32))
    n_changed += 1
print(f'smoothed {n_changed} frames -> {DST}')
assert n_changed > 100
