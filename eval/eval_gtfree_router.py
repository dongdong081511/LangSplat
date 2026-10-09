#!/usr/bin/env python
"""EXP-062: GT-free 自适应路由判据 (无人工标注, 无测试 GT).

判据信号 (全部来自 SAM 伪类 + 特征统计, 部署时对新场景零标注可用):
  1. pseudo-retrieval: SAM 跨帧 IoU 匹配对 (A,i)~(B,j), query=F_A[i] 在 F_B 全段检索, top1==j* 比例
  2. matched-pair cos 分布: mean / p10 / frac<0.7 (视角方差存活度的 2D 代理, EXP-061 机理)

路由规则 (机理先验, 非逐场景调参): route→DINO iff
  DINO pseudo-retrieval 胜 CLIP 且 DINO cos-p10 >= 0.75 (一致性尾部不过深, EXP-061 knife=0.665 为病理锚点)
"""
import os, sys, itertools, argparse
import numpy as np

SCENES = {
    'teatime':        [('CLIP8',       'language_features_dim8'),
                       ('DINO32',      'language_features_dim32_dino')],
    'figurines':      [('CLIP24',      'language_features_dim24'),
                       ('DINO32',      'language_features_dim32_dino')],
    'waldo_kitchen':  [('CLIP8',       'language_features_dim8'),
                       ('DINO32',      'language_features_dim32_dino'),
                       ('DINO64nat',   'language_features_dim64_dino_native')],
    'ramen':          [('CLIP8',       'language_features_dim8'),
                       ('DINO32',      'language_features_dim32_dino')],
}
MIN_PX, IOU_TH, MAX_Q, COS_TH = 100, 0.3, 30, 0.7
SEED, MAX_PAIRS = 0, 120


def load_frame(scene, subdir, fr):
    F = np.load(os.path.join(scene, subdir, fr + '_f.npy')).astype(np.float32)
    S = np.load(os.path.join(scene, subdir, fr + '_s.npy')).astype(np.int64)
    S = np.where(S[-1] < 0, 0, S[-1])  # 最细级 (索引覆盖全部 _f 行), -1=未标注→0
    return F, S


def eval_teacher(scene, subdir):
    frames = sorted(f[:-6] for f in os.listdir(os.path.join(scene, subdir)) if f.endswith('_f.npy'))
    data = {fr: load_frame(scene, subdir, fr) for fr in frames}
    rng = np.random.RandomState(SEED)
    all_pairs = list(itertools.combinations(frames, 2))
    if len(all_pairs) > MAX_PAIRS:
        all_pairs = [all_pairs[k] for k in rng.choice(len(all_pairs), MAX_PAIRS, replace=False)]
    hits1 = hits5 = n = 0
    coss = []
    for fa, fb in all_pairs:
        FA, SA = data[fa]; FB, SB = data[fb]
        cA = np.bincount(SA.ravel()); cB = np.bincount(SB.ravel())
        vA = [i for i in range(1, len(cA)) if cA[i] >= MIN_PX]
        vB = [j for j in range(1, len(cB)) if cB[j] >= MIN_PX]
        if not vA or not vB:
            continue
        vA = list(rng.permutation(vA))[:MAX_Q]  # 随机采样 (均匀覆盖大小段, 非偏置大段)
        KB = SB.max() + 1
        inter = np.bincount((SA * KB + SB).ravel(), minlength=len(cA) * KB).reshape(len(cA), KB)
        inter = inter[vA][:, vB]
        iou = inter / (cA[vA][:, None] + cB[vB][None, :] - inter + 1e-9)
        vBn = np.array(vB)
        FBv = FB[vB]
        FBv = FBv / (np.linalg.norm(FBv, axis=1, keepdims=True) + 1e-8)
        for row, i in enumerate(vA):
            jcol = int(np.argmax(iou[row]))
            if iou[row, jcol] < IOU_TH:
                continue
            q = FA[i] / (np.linalg.norm(FA[i]) + 1e-8)
            sim = FBv @ q
            top5 = np.argsort(-sim)[:5]
            hits1 += int(iou[row, top5[0]] >= IOU_TH)   # 物体级容错: top1 与匹配段 IoU>=0.3
            hits5 += int(iou[row, top5].max() >= IOU_TH)
            coss.append(float(q @ FBv[jcol]))
            n += 1
    coss = np.array(coss)
    return dict(n=n, top1=hits1 / max(n, 1), top5=hits5 / max(n, 1),
                cos_mean=float(coss.mean()) if n else 0,
                cos_p10=float(np.percentile(coss, 10)) if n else 0,
                frac_lo=float((coss < COS_TH).mean()) if n else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../dataset/lerf_ovs')
    args = ap.parse_args()
    res = {}
    for scene, teachers in SCENES.items():
        print(f'\n=== {scene}')
        res[scene] = {}
        for tname, subdir in teachers:
            r = eval_teacher(os.path.join(args.root, scene), subdir)
            res[scene][tname] = r
            print(f'  {tname:<10} n={r["n"]:>3}  top1={r["top1"]:.2%}  top5={r["top5"]:.2%}  '
                  f'cos_mean={r["cos_mean"]:.4f}  cos_p10={r["cos_p10"]:.4f}  frac<0.7={r["frac_lo"]:.2%}')
    # 路由决策 (机理先验规则): DINO 胜检索 且 cos_p10>=0.75
    print('\n=== routing (rule: DINO wins pseudo-retrieval AND DINO cos_p10>=0.75)')
    for scene, ts in res.items():
        dino = {k: v for k, v in ts.items() if k.startswith('DINO')}
        clip = {k: v for k, v in ts.items() if k.startswith('CLIP')}
        best_dino = max(dino.items(), key=lambda kv: kv[1]['top1'])
        best_clip = max(clip.items(), key=lambda kv: kv[1]['top1'])
        route = 'DINO' if (best_dino[1]['top1'] > best_clip[1]['top1'] and
                           best_dino[1]['cos_p10'] >= 0.75) else 'CLIP'
        expect = 'CLIP' if scene == 'waldo_kitchen' else 'DINO'
        print(f'  {scene:<15} dino_top1={best_dino[1]["top1"]:.2%} clip_top1={best_clip[1]["top1"]:.2%} '
              f'dino_p10={best_dino[1]["cos_p10"]:.4f}  -> route={route}  (exp054 winner={expect}) '
              f'{"OK" if route == expect else "MISS"}')


if __name__ == '__main__':
    main()
