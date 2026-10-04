"""Render fidelity: does the 3D field faithfully reproduce 2D tile features?"""
import os, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fidelity(scene, out_base, qdir, n_frames=3):
    segs = sorted(glob.glob(f'{ROOT}/dataset/lerf_ovs/{scene}/{qdir}/*_s.npy'))[:n_frames]
    out = []
    for s_p in segs:
        b = os.path.basename(s_p).replace('_s.npy', '')
        idx = int(b.split('_')[-1]) - 1
        seg = np.load(s_p).astype(np.int64)          # [4,H,W]
        f = np.load(s_p.replace('_s.npy', '_f.npy')).astype(np.float32)  # [T,32]
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        r_p = os.path.join(ROOT, 'output', f'{out_base}_1', 'train', 'ours_None',
                           'renders_npy', f'{idx:05d}.npy')
        if not os.path.exists(r_p):
            continue
        R = np.load(r_p).astype(np.float32)          # [H,W,32]
        Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
        cos = []
        for t in range(f.shape[0]):
            m = (seg[2] == t)                        # level-2 seg
            if m.sum() < 50:
                continue
            rm = Rn[m].mean(axis=0)
            rm = rm / (np.linalg.norm(rm) + 1e-8)
            cos.append(float(rm @ fn[t]))
        out.append((b, np.mean(cos), len(cos)))
    return out

for scene, ob, qd in [('teatime', 'teatime_dino_32d', 'language_features_dim32_dino'),
                      ('figurines', 'figurines_dino_32d', 'language_features_dim32_dino')]:
    res = fidelity(scene, ob, qd)
    vals = [v for _, v, _ in res]
    print(f'[render fidelity] {scene}: mean cos={np.mean(vals):.4f} ' +
          ' '.join(f'{b}:{v:.3f}({n})' for b, v, n in res))
