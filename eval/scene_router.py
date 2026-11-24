#!/usr/bin/env python
"""EXP-071: 场景级 GT-free 双信号路由器 (物体级判据 EXP-062a/b 失败后的场景级重试).

与 EXP-062 的本质区别: 只需回答"这个场景属于哪类" (场景级粗统计), 不需预测逐物体 3D 传导.
信号全部来自 CLIP 管线 SAM 段特征 (部署时对新场景零标注可用):
  Signal A dup_ratio: 帧内段特征 cos 聚类 (union-find, 阈值 τc) → 排除最大簇 (背景) 后,
                      多成员簇 (≥2 段, 语义重复物体如 knife×9) 的段占比
  Signal B n_cluster: 排除背景后的语义簇数/帧 (物体密度代理)
路由: dup_ratio > τ1 → CLIP; 否则 n_cluster < τ2 → CLIP; 否则 → DINO
协议: τ 在 4 主场景 (teatime/figurines/ramen/waldo) 上调, room/lawn/snacks 留出验证;
      batch2 的 6 场景训练完后作为第二留出集.
"""
import os
import sys
import glob
import argparse
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DS = os.path.join(BASE, 'dataset', 'lerf_ovs')
MIN_PX = 100  # 有效段最小像素 (沿用 EXP-062a)

# 留出集与主场景的 GT 胜者 (来自 EXP-039/040/070 的 GT 评测矩阵)
# EXP-072 扩展: batch2/3 六新场景的 image-query GT 对照 (mcnemar JSON 2026-10-15)
#   bed/bench/blue_sofa/table = 平局 (100/100), office_desk = CLIP (-3.33pp ns),
#   covered_desk = DINO (+8.00pp ns, n=25 b=0 c=2)
GT_WINNER = {'teatime': 'DINO', 'figurines': 'DINO', 'ramen': 'DINO',
             'waldo_kitchen': 'CLIP', 'room': 'CLIP', 'lawn': 'CLIP', 'snacks': 'CLIP',
             'bed': 'CLIP', 'bench': 'CLIP', 'blue_sofa': 'CLIP', 'table': 'CLIP',
             'office_desk': 'CLIP', 'covered_desk': 'DINO'}
TRAIN = ['teatime', 'figurines', 'ramen', 'waldo_kitchen']
HELDOUT = ['room', 'lawn', 'snacks', 'bed', 'bench', 'blue_sofa', 'table',
           'office_desk', 'covered_desk']


def union_find_clusters(F, cos_th):
    """F: (N, d) 已归一化段特征. 返回 labels (N,), 按簇成员数降序重排 (0=最大簇)."""
    N = len(F)
    if N == 0:
        return np.zeros(0, np.int64)
    sim = F @ F.T
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu, ju = np.where(np.triu(sim, 1) > cos_th)
    for a, b in zip(iu, ju):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    lab = np.array([find(i) for i in range(N)])
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)  # 0 = 最大簇 (背景)
    remap = np.empty(len(order), np.int64)
    remap[order] = np.arange(len(order))
    return remap[inv]


def scene_stats(scene, cos_th, feat='clip'):
    sub = os.path.join(DS, scene, 'language_features' if feat == 'clip' else 'language_features_dino')
    frames = sorted(f[:-6] for f in os.listdir(sub) if f.endswith('_f.npy'))
    n_cl, dup_r, dup_sz, covs = [], [], [], []
    for fr in frames:
        F = np.load(os.path.join(sub, fr + '_f.npy')).astype(np.float32)
        S = np.load(os.path.join(sub, fr + '_s.npy')).astype(np.int64)[-1]
        S = np.where(S < 0, 0, S)  # 未标注像素 (-1) → id 0 (同 EXP-062a)
        cnt = np.bincount(S.ravel(), minlength=len(F) + 1)
        valid = [i for i in range(1, len(F)) if cnt[i] >= MIN_PX]  # 0=id -1 映射/背景? 保守从 1 起
        if len(valid) < 3:
            continue
        covs.append(cnt[valid].sum() / S.size)  # 物体像素覆盖率 (信号 B 候选)
        if len(valid) < 3:
            continue
        Fv = F[valid]
        Fv = Fv / (np.linalg.norm(Fv, axis=1, keepdims=True) + 1e-8)
        lab = union_find_clusters(Fv, cos_th)
        K = lab.max() + 1
        sizes = [int((lab == k).sum()) for k in range(K)]
        n_cl.append(K - 1)  # 去掉最大簇
        objs = sizes[1:]    # 非背景簇
        dup_members = sum(s for s in objs if s >= 2)
        dup_r.append(dup_members / max(sum(objs), 1))
        dup_sz.append(max(objs) if objs else 0)
    return (len(frames), float(np.mean(n_cl)), float(np.mean(dup_r)),
            float(np.mean(dup_sz)) if dup_sz else 0.0, float(np.mean(covs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cos_th', type=float, default=0.90)
    ap.add_argument('--feat', choices=['clip', 'dino'], default='clip')
    ap.add_argument('--sweep', action='store_true', help='打印 τc 扫描表')
    ap.add_argument('--scenes', nargs='*', default=TRAIN + HELDOUT)
    args = ap.parse_args()

    cands = args.scenes if not args.sweep else [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    if args.sweep:
        print(f'τc sweep [feat={args.feat}] (master scenes: waldo dup_ratio 应显著高于其余三者):')
        print(f'{"scene":16s} ' + ' '.join(f'{t:>14.2f}' for t in cands))
        for sc in TRAIN:
            row = []
            for t in cands:
                _, nc, dr, _, _ = scene_stats(sc, t, args.feat)
                row.append(f'{dr:.3f}/{nc:5.1f}')
            print(f'{sc:16s} ' + ' '.join(f'{v:>14s}' for v in row))
        return

    print(f'{"scene":16s} {"frames":>6s} {"n_cl":>7s} {"dup_ratio":>9s} {"max_dup":>7s} {"cover":>6s} '
          f'{"GT胜者":>6s}')
    stats = {}
    for sc in args.scenes:
        nf, nc, dr, ds, cov = scene_stats(sc, args.cos_th, args.feat)
        stats[sc] = (nc, dr, cov)
        gt = GT_WINNER.get(sc, '?')
        print(f'{sc:16s} {nf:6d} {nc:7.1f} {dr:9.3f} {ds:7.1f} {cov:6.3f} {gt:>6s}', end='')
        if sc in TRAIN:
            print('  (训练集)')
        else:
            print('  (留出)')
    # τ1: waldo 与主场景其余三者的 dup_ratio 中点 (DINO 特征空间, τc=0.85)
    # τ2: 0.75 × 主场景最小 n_cl (留出场景不参与标定; 决策不变区间 (留出max, 主场景min); EXP-072 十场景留出后 = (7.8,13.6))
    if all(s in stats for s in TRAIN):
        dr_w = stats['waldo_kitchen'][1]
        dr_o = max(stats[s][1] for s in TRAIN if s != 'waldo_kitchen')
        nc_min = min(stats[s][0] for s in TRAIN)
        t1, t2 = (dr_w + dr_o) / 2, 0.75 * nc_min
        print(f'\nτ1={t1:.3f} (waldo {dr_w:.3f} vs 其余主场景峰 {dr_o:.3f}); '
              f'τ2={t2:.1f} (0.75×主场景最小 n_cl {nc_min:.1f}; 决策不变区间 ({max(stats[s][0] for s in args.scenes if s not in TRAIN):.1f}, {nc_min:.1f}))')
        n_ok = 0
        for sc in args.scenes:
            nc, dr, cov = stats[sc]
            route = 'CLIP' if (dr > t1 or nc < t2) else 'DINO'
            gt = GT_WINNER.get(sc, '?')
            ok = route == gt
            n_ok += ok
            print(f'  {sc:16s} → {route} (期望 {gt}) {"✓" if ok else "✗"}')
        print(f'路由一致: {n_ok}/{len(args.scenes)}')


if __name__ == '__main__':
    main()
