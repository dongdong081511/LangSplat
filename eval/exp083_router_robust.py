# exp083_router_robust.py — E2 路由泛化性 (审稿 B 维度: 阈值域迁移 + 信号扰动稳定)
#
# (a) leave-one-scene-out 阈值校准: 每次留出 1 个校准场景, 用其余 3 个重校 τ1/τ2,
#     检查 14 场景路由决策相对部署配置的翻转. τ 规则与部署协议同构:
#       τ2 = 0.75 × min n_cl(校准场景);  τ1 = 校准场景 dup 最大两值的中点
#     (全量 4 场景时该通用规则精确复现部署值 τ2=10.2, τ1=0.040).
# (b) 信号扰动稳定性: 校准场景帧子采样 (25%/50%, 10 seeds) 重算 Signal A/B,
#     报告信号漂移 + 每次扰动下"重校准→14 场景路由"相对部署决策的不变率.
#     已验证: DINOv2 特征 τc=0.85 逐位复现存档信号 (teatime 20.5/0.028 等).
# 用法: python eval/exp083_router_robust.py 2>&1 | tee eval_result/exp083_router_robust.log
import os
import sys
import importlib.util
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
spec = importlib.util.spec_from_file_location('sr', os.path.join(BASE, 'eval', 'scene_router.py'))
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

# 14 场景存档信号 (scene_router 实测, DINOv2 空间 τc=0.85; = exp081 SCENES 表)
SCENES = [
    ('teatime',       20.5, 0.028, 'DINO'),
    ('figurines',     21.4, 0.016, 'DINO'),
    ('ramen',         13.6, 0.010, 'DINO'),
    ('waldo_kitchen', 18.6, 0.052, 'CLIP'),
    ('room',           7.0, 0.000, 'CLIP'),
    ('lawn',           4.8, 0.000, 'CLIP'),
    ('snacks',         6.0, 0.000, 'CLIP'),
    ('sofa',           6.2, 0.000, 'tie'),
    ('bed',            5.0, 0.000, 'tie'),
    ('bench',          5.9, 0.000, 'CLIP'),
    ('blue_sofa',      5.0, 0.011, 'tie'),
    ('table',          7.8, 0.000, 'tie'),
    ('office_desk',    4.7, 0.000, 'CLIP'),
    ('covered_desk',   5.9, 0.000, 'DINO*'),
]
NAMES = [s[0] for s in SCENES]
N_CL = np.array([s[1] for s in SCENES])
DUP = np.array([s[2] for s in SCENES])
GT = {s[0]: s[3] for s in SCENES}
TRAIN = sr.TRAIN


def calibrate(names):
    """校准规则 (与部署协议同构): τ2 = 0.75×min n_cl; τ1 = top-2 dup 中点."""
    idx = [NAMES.index(n) for n in names]
    t2 = 0.75 * N_CL[idx].min()
    d = np.sort(DUP[idx])[::-1]
    t1 = (d[0] + d[1]) / 2
    return t2, t1


def route_all(t2, t1):
    return {n: ('CLIP' if (N_CL[i] < t2 or DUP[i] > t1) else 'DINO')
            for i, n in enumerate(NAMES)}


def acc(pred):
    ok = sum((GT[n] == 'tie') or (pred[n] == GT[n]) for n in NAMES)
    return ok


def part_a():
    print('=== E2(a) leave-one-scene-out 阈值校准 ===')
    t2_d, t1_d = calibrate(TRAIN)
    deployed = route_all(t2_d, t1_d)
    print(f'部署配置 (4 校准场景): τ2={t2_d:.2f} τ1={t1_d:.3f} → 判对 {acc(deployed)}/14')
    total_flips = 0
    for lo in TRAIN:
        rest = [s for s in TRAIN if s != lo]
        t2, t1 = calibrate(rest)
        pred = route_all(t2, t1)
        flips = [n for n in NAMES if pred[n] != deployed[n]]
        total_flips += len(flips)
        print(f'  留出 {lo:14s}: τ2={t2:5.2f} τ1={t1:.3f}  判对 {acc(pred)}/14  '
              f'翻转 {len(flips)}{": " + ",".join(flips) if flips else ""}')
    print(f'  LOO 平均决策翻转 {total_flips / len(TRAIN):.2f}/14 '
          f'→ 决策不变率 {100 * (1 - total_flips / (len(TRAIN) * 14)):.1f}%')

    # Otsu 自动阈值的 LOO 变体 (exp081 auto 配方: τ2 全局 Otsu; τ1 密集子集条件 Otsu)
    print('\n  --- Otsu 自动阈值 LOO (exp081 hier-Otsu 配方) ---')
    def otsu_1d(x):
        xs = np.sort(np.unique(x))
        best_t, best_v = None, -1
        for i in range(len(xs) - 1):
            t = (xs[i] + xs[i + 1]) / 2
            lo_, hi_ = x[x <= t], x[x > t]
            if len(lo_) == 0 or len(hi_) == 0:
                continue
            w0, w1 = len(lo_) / len(x), len(hi_) / len(x)
            v = w0 * w1 * (lo_.mean() - hi_.mean()) ** 2
            if v > best_v:
                best_t, best_v = t, v
        return best_t

    def otsu_pred(keep):
        idx = [i for i, n in enumerate(NAMES) if n not in keep]
        t2 = otsu_1d(N_CL[idx])
        dense = N_CL[idx] > t2
        t1 = otsu_1d(DUP[idx][dense]) if dense.sum() >= 2 else 0.0
        return t2, t1

    t2_o, t1_o = otsu_pred(set())
    base_o = route_all(t2_o, t1_o)
    print(f'  全量 Otsu: τ2={t2_o:.2f} τ1={t1_o:.3f} → 判对 {acc(base_o)}/14')
    total_flips = 0
    for lo in NAMES:
        t2, t1 = otsu_pred({lo})
        pred = route_all(t2, t1)
        flips = [n for n in NAMES if pred[n] != base_o[n]]
        total_flips += len(flips)
        if flips or lo in TRAIN:
            print(f'  留出 {lo:14s}: τ2={t2:5.2f} τ1={t1:.3f}  判对 {acc(pred)}/14  '
                  f'翻转 {len(flips)}{": " + ",".join(flips) if flips else ""}')
    print(f'  Otsu-LOO 平均决策翻转 {total_flips / len(NAMES):.2f}/14 '
          f'→ 决策不变率 {100 * (1 - total_flips / (len(NAMES) * 14)):.1f}%')


def scene_stats_frames(scene, cos_th, frames):
    """scene_router.scene_stats 的子采样版: 只统计给定帧."""
    sub = os.path.join(sr.DS, scene, 'language_features_dino')
    n_cl, dup_r = [], []
    for fr in frames:
        F = np.load(os.path.join(sub, fr + '_f.npy')).astype(np.float32)
        S = np.load(os.path.join(sub, fr + '_s.npy')).astype(np.int64)[-1]
        S = np.where(S < 0, 0, S)
        cnt = np.bincount(S.ravel(), minlength=len(F) + 1)
        valid = [i for i in range(1, len(F)) if cnt[i] >= sr.MIN_PX]
        if len(valid) < 3:
            continue
        Fv = F[valid]
        Fv = Fv / (np.linalg.norm(Fv, axis=1, keepdims=True) + 1e-8)
        lab = sr.union_find_clusters(Fv, cos_th)
        K = lab.max() + 1
        sizes = [int((lab == k).sum()) for k in range(K)]
        n_cl.append(K - 1)
        objs = sizes[1:]
        dup_members = sum(s for s in objs if s >= 2)
        dup_r.append(dup_members / max(sum(objs), 1))
    return float(np.mean(n_cl)), float(np.mean(dup_r))


def all_frames(scene):
    sub = os.path.join(sr.DS, scene, 'language_features_dino')
    return sorted(f[:-6] for f in os.listdir(sub) if f.endswith('_f.npy'))


def part_b(keeps=(0.25, 0.50), seeds=range(10), cos_th=0.85):
    print('\n=== E2(b) 信号扰动稳定性 (校准场景帧子采样, DINOv2 τc=0.85) ===')
    t2_d, t1_d = calibrate(TRAIN)
    deployed = route_all(t2_d, t1_d)
    frames = {sc: all_frames(sc) for sc in TRAIN}
    # 全量重算 (验证口径) + 子采样
    full_sig = {}
    for sc in TRAIN:
        nc, dr = scene_stats_frames(sc, cos_th, frames[sc])
        full_sig[sc] = (nc, dr)
        print(f'  {sc:16s} 全量: n_cl={nc:.1f} dup={dr:.3f} (frames={len(frames[sc])})')
    print(f'  全量重算 vs 存档: 最大偏差 n_cl {max(abs(full_sig[s][0] - N_CL[NAMES.index(s)]) for s in TRAIN):.2f}, '
          f'dup {max(abs(full_sig[s][1] - DUP[NAMES.index(s)]) for s in TRAIN):.4f}')
    for keep in keeps:
        print(f'  --- keep={keep:.0%}, {len(list(seeds))} seeds ---')
        drift = {sc: [] for sc in TRAIN}
        flip_total = decisions = 0
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sig = {}
            for sc in TRAIN:
                k = max(1, int(round(keep * len(frames[sc]))))
                fr = list(rng.choice(frames[sc], k, replace=False))
                sig[sc] = scene_stats_frames(sc, cos_th, fr)
                drift[sc].append(sig[sc])
            # 扰动信号下重校准 → 14 场景路由 (留出场景信号为存档值, 扰动仅通过 τ 传导)
            idx = [NAMES.index(s) for s in TRAIN]
            N_CL[idx] = [sig[s][0] for s in TRAIN]
            DUP[idx] = [sig[s][1] for s in TRAIN]
            t2, t1 = calibrate(TRAIN)
            pred = route_all(t2, t1)
            flip_total += sum(pred[n] != deployed[n] for n in NAMES)
            decisions += len(NAMES)
            N_CL[idx] = [s[1] for s in SCENES[:4]]
            DUP[idx] = [s[2] for s in SCENES[:4]]
        for sc in TRAIN:
            arr = np.array(drift[sc])
            f0 = full_sig[sc]
            rn = np.abs(arr[:, 0] - f0[0]).max() / f0[0]
            rd = np.abs(arr[:, 1] - f0[1]).max() / max(f0[1], 1e-9)
            print(f'    {sc:16s} n_cl {arr[:,0].mean():6.1f}±{arr[:,0].std():4.1f} '
                  f'(max rel drift {100*rn:4.1f}%) | dup {arr[:,1].mean():.3f}±{arr[:,1].std():.3f} '
                  f'(max rel drift {100*rd:4.1f}%)')
        print(f'    14 场景决策不变率: {100 * (1 - flip_total / decisions):.1f}% '
              f'({flip_total} flips / {decisions} decisions)')


if __name__ == '__main__':
    part_a()
    part_b()
