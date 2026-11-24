"""EXP-057c: DINOv3 失败机理诊断 — 特征空间正/负样本分离度分析。

对每个 GT 对: query tile vs 同帧正样本 tiles / 负样本 tiles 的余弦相似度分布,
计算分离度 AUC (随机取一正一负, 正>负 的概率) 与 pos_sim std (EXP-025 判据)。
"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, '/home/xiedexia/project/LangSplat/eval')
from tile_retrieval_test import load_gt, pick_query_tile

ROOT = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs'
SCENE = 'waldo_kitchen'
SRC = {
    'CLIP': os.path.join(ROOT, SCENE, 'language_features'),
    'DINOv2': os.path.join(ROOT, SCENE, 'language_features_dino'),
    'DINOv3_mean224': os.path.join(ROOT, SCENE, 'dino3_rescue_m224'),
    'DINOv3_cls224': os.path.join(ROOT, SCENE, 'dino3_rescue_c224'),
}

gt = load_gt()
rng = np.random.default_rng(0)

for name, src in SRC.items():
    pos_all, neg_all = [], []
    for frame, objs in gt.items():
        fp = os.path.join(src, f'{frame}_f.npy')
        sp = os.path.join(src, f'{frame}_s.npy')
        if not os.path.exists(fp):
            continue
        F = np.load(fp).astype(np.float32)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        S = np.load(sp).astype(np.int64)
        for L in range(S.shape[0]):
            seg = S[L]
            for i, o in enumerate(objs):
                q = pick_query_tile(seg, o['mask'])
                if q is None:
                    continue
                pos = set()
                for j, o2 in enumerate(objs):
                    if o2['label'] == o['label']:
                        idxs = np.unique(seg[o2['mask'] > 0])
                        pos.update(int(t) for t in idxs if t >= 0)
                pos.discard(int(q))
                if not pos:
                    continue
                neg = [t for t in np.unique(seg) if t >= 0 and int(t) not in pos and t != q]
                sim = F @ F[q]
                pos_all += [float(sim[t]) for t in pos]
                neg_all += [float(sim[t]) for t in neg[:200]]
    pos, neg = np.array(pos_all), np.array(neg_all)
    n = min(len(pos), len(neg))
    # AUC 手算: 随机一正一负, P(pos>neg) + 0.5*P(tie)
    wins = (pos[:n, None] > neg[None, :n]).sum() + 0.5 * (pos[:n, None] == neg[None, :n]).sum()
    auc = wins / (n * n)
    print(f'{name:16s} pos mean={pos.mean():.3f} std={pos.std():.4f} | '
          f'neg mean={neg.mean():.3f} std={neg.std():.4f} | gap={pos.mean()-neg.mean():+.4f} | AUC={auc:.4f}')
