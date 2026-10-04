"""Cross-view consistency of tile features: same-object tiles across frames (CLIP vs DINO)."""
import os, sys, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_image_query import load_gt, pick_query_tile

def analyze(scene, gt_dir, clip_dir, dino_dir):
    gt = load_gt(gt_dir)
    frames = sorted(gt.keys())
    clip_segs = {fr: np.load(os.path.join(clip_dir, f'{fr}_s.npy')).astype(np.int64) for fr in frames}
    dino_segs = {fr: np.load(os.path.join(dino_dir, f'{fr}_s.npy')).astype(np.int64) for fr in frames}
    clip_f = {fr: np.load(os.path.join(clip_dir, f'{fr}_f.npy')).astype(np.float32) for fr in frames}
    dino_f = {fr: np.load(os.path.join(dino_dir, f'{fr}_f.npy')).astype(np.float32) for fr in frames}

    def norm(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    clip_f = {k: norm(v) for k, v in clip_f.items()}
    dino_f = {k: norm(v) for k, v in dino_f.items()}

    pairs = {'CLIP': [], 'DINO': []}
    for frA in frames:
        for obj in gt[frA]:
            for frB in frames:
                if frB == frA: continue
                for o2 in gt[frB]:
                    if o2['label'] != obj['label']: continue
                    qa = pick_query_tile(clip_segs[frA][0], obj['mask'])
                    qb = pick_query_tile(clip_segs[frB][0], o2['mask'])
                    da = pick_query_tile(dino_segs[frA][0], obj['mask'])
                    db = pick_query_tile(dino_segs[frB][0], o2['mask'])
                    if None in (qa, qb, da, db): continue
                    if qa >= clip_f[frA].shape[0] or qb >= clip_f[frB].shape[0]: continue
                    if da >= dino_f[frA].shape[0] or db >= dino_f[frB].shape[0]: continue
                    pairs['CLIP'].append(float(clip_f[frA][qa] @ clip_f[frB][qb]))
                    pairs['DINO'].append(float(dino_f[frA][da] @ dino_f[frB][db]))
    for k in ('CLIP', 'DINO'):
        v = pairs[k]
        print(f'[{scene}] {k}: cross-view same-object cos mean={np.mean(v):.4f} std={np.std(v):.4f} n={len(v)}')

base = os.path.join(ROOT, 'dataset', 'lerf_ovs')
analyze('teatime', f'{base}/label/teatime', f'{base}/teatime/language_features', f'{base}/teatime/language_features_dino')
analyze('figurines', f'{base}/label/figurines', f'{base}/figurines/language_features', f'{base}/figurines/language_features_dino')
