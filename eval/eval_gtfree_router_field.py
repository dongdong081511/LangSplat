#!/usr/bin/env python
"""EXP-062b: 场级 (渲染特征) GT-free 路由判据 — SAM 伪类检索 on renders_npy (配对协议 v2).

机理: 路由信号必须测"场交付了什么" (EXP-062a 证明 2D 统计无法预测 3D 传导失败).
GT-free: 只用 SAM mask (无人工标注, 无测试 GT); 部署时两场已训练, 路由发生在训练后/服务前.

v3 峰值检索协议 (镜像部署): v2 段均值检索把渲染场最锋利的 per-pixel 峰值结构抹掉, 两场打平.
v3 完全复刻 EXP-054 部署检索机制: query=帧A段掩码均值特征 → 帧B逐像素相似度 → 30×30 平滑
→ argmax 峰值点, hit=峰值落在匹配段 (IoU>=0.3 跨帧匹配) 掩码内; SAM 段替代 GT 框, 保持 GT-free.
渲染 idx = 第 i 个已存在帧 (按 image_name 排序, scene/dataset_readers.py L151);
帧号有缺口 (waldo 缺 155/156/157 等), 严禁用 frame 号-1 当渲染 idx.
"""
import os, itertools
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
P = lambda *a: os.path.join(ROOT, *a)

# CLIP 特征目录 = 统一分割源 + 其渲染场; DINO 场只换渲染目录 (a07, 判据测场判别度与部署 α 无关)
SCENES = {
    'teatime': dict(clip='teatime_8d', dino='teatime_dino_32ds_a07',
                    feat=P('dataset/lerf_ovs/teatime/language_features_dim8')),
    'figurines': dict(clip='figurines_24d', dino='figurines_dino_32ds_a07',
                      feat=P('dataset/lerf_ovs/figurines/language_features_dim24')),
    'waldo_kitchen': dict(clip='waldo_kitchen_clip8d', dino='waldo_kitchen_dino_32ds_a07',
                          feat=P('dataset/lerf_ovs/waldo_kitchen/language_features_dim8')),
    'ramen': dict(clip='ramen_clip8d', dino='ramen_dino_32ds_a07',
                  feat=P('dataset/lerf_ovs/ramen/language_features_dim8')),
}
GT_WINNER = {'teatime': 'DINO', 'figurines': 'DINO', 'waldo_kitchen': 'CLIP', 'ramen': 'DINO'}
MIN_PX, IOU_TH, MAX_Q, N_PAIRS, SEED = 100, 0.3, 30, 80, 0
SEGLVL = int(os.environ.get('SEGLVL', '0'))  # 0=最粗级(段大, 接近物体级) / 1=level-2 对应级(碎片多)


def seg_means(R, seg_ids, mask_flat, cnt):
    """给定段的掩码均值特征; seg_ids/mask_flat/cnt 由协议预计算."""
    flat = mask_flat
    K = int(flat.max()) + 1
    M = np.empty((K, R.shape[-1]), dtype=np.float64)
    for c in range(R.shape[-1]):
        M[:, c] = np.bincount(flat, weights=R[..., c].ravel(), minlength=K)
    return (M[seg_ids] / cnt[seg_ids][:, None]).astype(np.float32)


def build_protocol(feat_dir, rng):
    """统一 query 协议: 帧对采样 + IoU>=0.3 段匹配 + query 段采样, 全部只算一次."""
    frames = sorted(f[:-6] for f in os.listdir(feat_dir) if f.endswith('_s.npy'))
    pairs = list(itertools.combinations(range(len(frames)), 2))
    if len(pairs) > N_PAIRS:
        pairs = [pairs[k] for k in rng.choice(len(pairs), N_PAIRS, replace=False)]
    proto = []  # (ia, ib, [ (seg_a, jcol) ... ])
    for ia, ib in pairs:
        SA = np.load(os.path.join(feat_dir, frames[ia] + '_s.npy')).astype(np.int64)[SEGLVL]
        SB = np.load(os.path.join(feat_dir, frames[ib] + '_s.npy')).astype(np.int64)[SEGLVL]
        SA = np.where(SA < 0, 0, SA); SB = np.where(SB < 0, 0, SB)
        cA = np.bincount(SA.ravel()); cB = np.bincount(SB.ravel())
        vA = [i for i in range(1, len(cA)) if cA[i] >= MIN_PX]
        vB = [j for j in range(1, len(cB)) if cB[j] >= MIN_PX]
        if not vA or not vB:
            continue
        vA = list(rng.permutation(vA))[:MAX_Q]
        KB = SB.max() + 1
        inter = np.bincount((SA * KB + SB).ravel(), minlength=len(cA) * KB).reshape(len(cA), KB)
        iou = inter[vA][:, vB] / (cA[vA][:, None] + cB[vB][None, :] - inter[vA][:, vB] + 1e-9)
        rows = []
        for r, seg_a in enumerate(vA):
            jc = int(np.argmax(iou[r]))
            if iou[r, jc] >= IOU_TH:
                rows.append((int(seg_a), jc))  # jc = vB 内索引, 命中=峰值落在 vB[jc] 段掩码内
        if rows:
            proto.append((ia, ib, SA.astype(np.int32), SB.astype(np.int32),
                          np.array(cA), np.array(cB), vB, rows))
    return proto


def eval_field_paired(render_dir, proto):
    """v3 峰值检索: query=段掩码均值, DB=帧B逐像素相似度→30×30平滑→argmax, hit=峰在匹配段内."""
    rn = P('output', render_dir + '_2', 'train', 'ours_None', 'renders_npy')
    kernel = np.ones((30, 30)) / 900
    hits, coss = [], []
    for ia, ib, SA, SB, cA, cB, vB, rows in proto:
        RA = np.load(os.path.join(rn, f'{ia:05d}.npy')).astype(np.float32)
        RB = np.load(os.path.join(rn, f'{ib:05d}.npy')).astype(np.float32)
        RBn = (RB / (np.linalg.norm(RB, axis=-1, keepdims=True) + 1e-8)).reshape(-1, RB.shape[-1])
        seg_a = np.array([r[0] for r in rows])
        Q = seg_means(RA, seg_a, SA.ravel(), cA)        # (n_rows, C)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        sim = RBn @ Q.T                                  # (HW, n_rows)
        for qi, (seg_a_i, jc) in enumerate(rows):
            simf = cv2.filter2D(sim[:, qi].reshape(SB.shape), -1, kernel)
            pt = np.unravel_index(np.argmax(simf), simf.shape)
            hits.append(int(SB[pt] == vB[jc]))
            coss.append(float(simf[pt]))
    return np.array(hits), np.array(coss)


def main():
    rng = np.random.RandomState(SEED)
    all_ok = 0
    for scene, cfg in SCENES.items():
        print(f'\n=== {scene} (paired field-level pseudo-retrieval, level-2 renders)', flush=True)
        proto = build_protocol(cfg['feat'], rng)
        n_q = sum(len(p[7]) for p in proto)
        hc, cc = eval_field_paired(cfg['clip'], proto)
        hd, cd = eval_field_paired(cfg['dino'], proto)
        b = int(((hc == 0) & (hd == 1)).sum())  # CLIP-only wins
        c = int(((hc == 1) & (hd == 0)).sum())  # DINO-only wins
        route = 'DINO' if hd.mean() > hc.mean() else 'CLIP'
        good = route == GT_WINNER[scene]
        all_ok += good
        print(f'  n={n_q}  CLIP top1={hc.mean():.2%} (cos={cc.mean():.3f})  '
              f'DINO top1={hd.mean():.2%} (cos={cd.mean():.3f})', flush=True)
        print(f'  paired: b(clip-only)={b} c(dino-only)={c}  -> route {route} '
              f'(exp054={GT_WINNER[scene]})  {"OK" if good else "MISS"}', flush=True)
    print(f'\nrouting accuracy: {all_ok}/4', flush=True)


if __name__ == '__main__':
    main()
