"""EXP-057d: 区分 DINOv3 坍缩的两个竞争假说。
H1 池化污染: 黑背景 patch 主导 mean -> masked mean 应恢复判别性
H2 gram anchoring: 特征空间本身在 OOD 输入上同质 -> masked mean 仍坍缩
输出: full-mean / mask-mean / cls 三种池化的 pos/neg gap + AUC + 黑patch占比统计。
"""
import sys, os, glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, '/home/xiedexia/project/LangSplat')
sys.path.insert(0, '/home/xiedexia/project/LangSplat/eval')
from tile_retrieval_test import load_gt, pick_query_tile
from mm_langsplat.extractors.dinov3_extractor import DINOv3Extractor, _DINOV3_MEAN, _DINOV3_STD

ROOT = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs'
SCENE = 'waldo_kitchen'
SRC = os.path.join(ROOT, SCENE, 'language_features_dino3')
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
P, S = 16, 14  # patch size / grid (224)

_MEAN = torch.tensor(_DINOV3_MEAN).view(1, 3, 1, 1).cuda()
_STD = torch.tensor(_DINOV3_STD).view(1, 3, 1, 1).cuda()
gt = load_gt()

ex = DINOv3Extractor(verbose=False).cuda()
res_pool = {'full': [], 'masked': [], 'cls': []}
black_frac_all = []

segs_files = [os.path.join(SRC, f + '_s.npy') for f in list(gt.keys())[:N_FRAMES]]
for sf in segs_files:
    frame = os.path.basename(sf)[:-6]
    if frame not in gt:
        continue
    img = cv2.imread(os.path.join(ROOT, SCENE, 'images', frame + '.jpg'))
    segs = np.load(sf).astype(np.int64)
    ids = np.unique(segs); ids = ids[ids >= 0]
    tiles, masks_patch = [], []
    for t in ids:
        m = (segs == t).any(axis=0)
        ys, xs = np.nonzero(m)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tile = img[y1:y2, x1:x2].copy()
        tile[m[y1:y2, x1:x2] == 0] = 0
        tri = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        black_frac_all.append(float((m[y1:y2, x1:x2] == 0).mean()))
        # resize 到 224 后按 14x14 patch 统计黑色占比 (RGB 亮度)
        r = cv2.resize(tri, (224, 224)).astype(np.float32) / 255.0
        pm = r.mean(axis=2)  # [224,224]
        pmask = np.zeros((S, S), bool)
        for i in range(S):
            for j in range(S):
                pmask[i, j] = pm[i*P:(i+1)*P, j*P:(j+1)*P].mean() > 0.05  # 非黑 patch
        masks_patch.append(pmask)
        t = torch.from_numpy(r).permute(2, 0, 1)
        tiles.append(t)
    x = (torch.stack(tiles).cuda() - _MEAN) / _STD
    with torch.no_grad():
        out = ex.model.forward_features(x)
        d = out[0] if isinstance(out, (list, tuple)) else out
        pt = d['x_norm_patchtokens']          # [N, 196, D]
        cl = F.normalize(d['x_norm_clstoken'], dim=-1).cpu().numpy()
    full = F.normalize(pt.mean(dim=1), dim=-1).cpu().numpy()
    maskmean = np.zeros_like(full)
    for n in range(len(tiles)):
        sel = torch.from_numpy(masks_patch[n]).flatten().cuda()
        if sel.sum() == 0:
            sel[:] = True
        maskmean[n] = F.normalize(pt[n][sel].mean(dim=0), dim=0).cpu().numpy()
    res_pool['full'].append(full)
    res_pool['masked'].append(maskmean)
    res_pool['cls'].append(cl)

pool = {k: np.concatenate(v) for k, v in res_pool.items()}
print(f'tiles={len(pool["full"])}  mean black_frac={np.mean(black_frac_all):.2%}  median={np.median(black_frac_all):.2%}')

# 相似度结构诊断
for k, X in pool.items():
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    off = sim[np.triu_indices(len(sim), 1)]
    print(f'{k:8s} 全体 tile 间 cos: mean={off.mean():.4f} p5={np.percentile(off,5):.4f} p95={np.percentile(off,95):.4f}')

# GT 对 pos/neg gap + AUC (按帧偏移切片)
n_per_frame = [len(v) for v in res_pool['full']]
offs = np.cumsum([0] + n_per_frame)
for k, X in pool.items():
    pos_all, neg_all = [], []
    for fi, (frame, objs) in enumerate(gt.items()):
        if fi >= N_FRAMES:
            break
        Fm = X[offs[fi]:offs[fi + 1]]
        Fm = Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-8)
        segs = np.load(os.path.join(SRC, frame + '_s.npy')).astype(np.int64)
        for L in range(segs.shape[0]):
            seg = segs[L]
            for o in objs:
                q = pick_query_tile(seg, o['mask'])
                if q is None or q >= len(Fm):
                    continue
                pos = set()
                for o2 in objs:
                    if o2['label'] == o['label']:
                        pos.update(int(t) for t in np.unique(seg[o2['mask'] > 0]) if t >= 0)
                pos.discard(int(q))
                if not pos:
                    continue
                neg = [int(t) for t in np.unique(seg) if t >= 0 and int(t) not in pos and t != q]
                sim = Fm @ Fm[q]
                pos_all += [float(sim[t]) for t in pos if t < len(sim)]
                neg_all += [float(sim[t]) for t in neg[:200] if t < len(sim)]
    pos, neg = np.array(pos_all), np.array(neg_all)
    n = min(len(pos), len(neg))
    wins = (pos[:n, None] > neg[None, :n]).sum() + 0.5 * (pos[:n, None] == neg[None, :n]).sum()
    print(f'{k:8s} pos mean={pos.mean():.3f} neg mean={neg.mean():.3f} gap={pos.mean()-neg.mean():+.4f} AUC={wins/(n*n):.4f}')
