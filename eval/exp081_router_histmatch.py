# C3 (审稿修改): 路由阈值的域自适应重校 — GT-free 直方图分割 (1D Otsu / 2-means)
#
# 动机 (CVPR_review C3): τ1/τ2 目前是源域上手工定的中点值, 审稿人要求提供
# "domain-adaptive thresholding (even simple histogram matching)" 以支撑 GT-free 部署.
# 本实验把阈值选择本身做成无监督过程: 对场景级信号分布做 1D 二分割, 自动找分离点,
# 对照手工阈值与 GT 胜者判决.
#
# 数据: hyper_parameter.md EXP-071 13 场景 + Tab.8 sofa (n_cl=6.2, dup=0.000, tie)
# (scene_router.py 实测, DINOv2 空间 τc=0.85; 14 场景全量口径 2026-11-04 更新)
# GT 期望: image eval McNemar 显著胜者; ns 差异记"平" (路由到 CLIP = 安全回退).
#
# 用法: python eval/exp081_router_histmatch.py   (纯 CPU, 秒级)
import numpy as np

# (场景, n_cl, dup_ratio, GT 期望)  — EXP-071 归档 L1385-1392 + Tab.8 sofa 行
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
    ('covered_desk',   5.9, 0.000, 'DINO*'),  # +8pp 但 ns, 部署代价上界 8pp
]

def otsu_1d(x):
    """1D Otsu: 最优类间方差分割点 (返回阈值=相邻类均值中点)."""
    xs = np.sort(np.unique(x))
    best_t, best_v = None, -1
    for i in range(len(xs) - 1):
        t = (xs[i] + xs[i + 1]) / 2
        lo, hi = x[x <= t], x[x > t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w0, w1 = len(lo) / len(x), len(hi) / len(x)
        v = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if v > best_v:
            best_t, best_v = t, v
    return best_t

def kmeans_1d(x, seed=0, tries=50):
    """1D 2-means (多初始质心防局部最优), 返回两质心中点为阈值."""
    rng = np.random.default_rng(seed)
    best_t, best_inertia = None, np.inf
    for _ in range(tries):
        c = rng.choice(x, 2, replace=False)
        for _ in range(100):
            lo, hi = x[x <= c.mean()], x[x > c.mean()]
            if len(lo) == 0 or len(hi) == 0:
                break
            new = np.array([lo.mean(), hi.mean()])
            if np.allclose(new, c, atol=1e-6):
                break
            c = new
        inertia = ((x[x <= c.mean()] - c[0]) ** 2).sum() + ((x[x > c.mean()] - c[1]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_t = inertia, c.mean()
    return best_t

def route(n_cl, dup, t2, t1):
    """路由规则 (§3.4): n_cl < τ2 → CLIP; 否则 dup > τ1 → CLIP; 否则 DINO."""
    if n_cl < t2 or dup > t1:
        return 'CLIP'
    return 'DINO'

def score(t2, t1, tag):
    n_ok, details = 0, []
    for name, n, d, gt in SCENES:
        pred = route(n, d, t2, t1)
        # 判定: tie 场景路由到任一方都算 ✓ (零差异); DINO* 的 CLIP 判决算 ✗ (审稿口径)
        ok = (gt == 'tie') or (pred == gt)
        n_ok += ok
        details.append((name, pred, gt, '✓' if ok else '✗'))
    print(f'--- {tag}: τ2={t2:.3f} τ1={t1:.3f} → {n_ok}/{len(SCENES)}')
    for name, pred, gt, mark in details:
        if mark == '✗' or gt == 'DINO*':
            print(f'    {name:<14} pred={pred:<4} gt={gt} {mark}')
    return n_ok

print('=== 手工阈值基线 (τ2=10.2, τ1=0.040, 源域中点) ===')
n_manual = score(10.2, 0.040, 'manual')
print()
for sig, signame, manual in [(np.array([s[1] for s in SCENES]), 'n_cl (Signal B)', 10.2),
                             (np.array([s[2] for s in SCENES]), 'dup_ratio (Signal A)', 0.040)]:
    t_otsu, t_km = otsu_1d(sig), kmeans_1d(sig)
    print(f'=== {signame}: Otsu τ*={t_otsu:.3f} | 2-means τ*={t_km:.3f} (manual {manual})')
print()
n_otsu = score(otsu_1d(np.array([s[1] for s in SCENES])),
               otsu_1d(np.array([s[2] for s in SCENES])), 'auto (Otsu)')
n_km = score(kmeans_1d(np.array([s[1] for s in SCENES])),
             kmeans_1d(np.array([s[2] for s in SCENES])), 'auto (2-means)')
print()
# === 层级重校 (与路由两阶段结构同构): Signal B 全局; Signal A 条件化于密集子集 ===
# 动机: 全局 Otsu 的 τ1 被九个零冗余场景拉低 (0.022), teatime 误判 CLIP;
# ρ_dup 信号只在 n_cl > τ2* 的密集子集 (4 LERF 场景) 上有语义, τ1 应在该子集上重校.
N_CL = np.array([s[1] for s in SCENES])
DUP = np.array([s[2] for s in SCENES])
for t2_star, tag_t2 in [(otsu_1d(N_CL), 'Otsu'), (kmeans_1d(N_CL), '2-means')]:
    dense = N_CL > t2_star
    t1_h_otsu = otsu_1d(DUP[dense])
    t1_h_km = kmeans_1d(DUP[dense])
    print(f'=== Hierarchical ({tag_t2}): τ2={t2_star:.3f}; dense subset = '
          f'{[SCENES[i][0] for i in range(len(SCENES)) if dense[i]]}')
    score(t2_star, t1_h_otsu, f'hier-{tag_t2} Otsu-τ1')
    score(t2_star, t1_h_km, f'hier-{tag_t2} 2means-τ1')
print()
# 决策不变区间检验: 自动阈值是否落入手工阈值的决策不变区间
print(f'决策不变区间: τ2 ∈ (7.8, 13.6), τ1 ∈ (0.028, 0.052)  [相邻判决翻转的信号值界]')
print(f'手工基线 {n_manual}/{len(SCENES)} | 全局 Otsu {n_otsu}/{len(SCENES)} | 全局 2-means {n_km}/{len(SCENES)}')
