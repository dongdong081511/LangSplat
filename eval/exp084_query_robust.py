# exp084_query_robust.py — E3 查询条件多样性 (审稿 C 维度: query 帧遍历 + 查询扰动鲁棒性)
#
# 基于 rm_query_compare.py 的穷举 GT 帧对协议 (RM: query=帧A GT mask 内渲染特征均值,
# DB=帧B 渲染特征余弦图 → 30×30 均值滤波 → argmax, hit=峰值落同名 GT bbox).
#
# (a) query 帧遍历: 每个查询事件按查询帧分组 → per-query-frame 准确率分布
#     (mean/std/min/max) + 查询事件级 bootstrap 95% CI (1000 次).
# (b) 查询扰动鲁棒性: 仅扰动查询构造 (DB 不变), 三族:
#       mask 形变:  膨胀/腐蚀 ±5/10/20% sqrt(area), 以及 bbox 矩形替代多边形 mask
#       特征噪声:   q' = normalize(q + σ·ε), σ ∈ {0.05,0.1,0.2,0.4}
#       背景混入:   q' = normalize((1-α)q + α·q_ring), α ∈ {0.1,0.25,0.5}
#     场景: bed/table (唯一同时有 CLIP 8d 与 DINOv2 32d 渲染的场景对).
# 用法: python eval/exp084_query_robust.py 2>&1 | tee eval_result/exp084_query_robust.log
import os
import glob
import json
import numpy as np
import cv2

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DS = os.path.join(BASE, 'dataset', 'lerf_ovs')
OUT = os.path.join(BASE, 'output')
KERNEL = np.ones((30, 30)) / 900
MIN_PX = 30

FIELDS = {
    # 双场对照 (CLIP vs DINO)
    'bed':           [('CLIP 8d', 'bed_clip8d_1'), ('DINOv2 32d', 'bed_dino_32ds_a07_1')],
    'table':         [('CLIP 8d', 'table_clip8d_1'), ('DINOv2 32d', 'table_dino_32ds_a07_1')],
    'ramen':         [('CLIP 8d', 'ramen_clip8d_1'), ('DINOv2 32d', 'ramen_dino_32ds_a07_1')],
    'waldo_kitchen': [('CLIP 8d', 'waldo_kitchen_clip8d_1'), ('DINOv2 32d', 'waldo_kitchen_dino_32ds_a07_1')],
    # 单场 (仅部署场有渲染)
    'bench':         [('CLIP 8d', 'bench_clip8d_1')],
    'room':          [('CLIP 8d', 'room_clip8d_1')],
    'lawn':          [('CLIP 8d', 'lawn_clip8d_1')],
    'sofa':          [('CLIP 8d', 'sofa_clip8d_1')],
    'teatime':       [('DINOv2 32d', 'teatime_dino_32ds_a07_1')],
}
# LERF 帧 frame_00002 (1-based) → renders 00001; OVS 02 → 00002
render_idx = lambda name: (int(name[6:].split('.')[0]) - 1 if name.startswith('frame_')
                           else int(name.split('.')[0]))
NOISE_SIGMAS = [0.05, 0.1, 0.2, 0.4]
MIX_ALPHAS = [0.1, 0.25, 0.5]
JITTER_PCTS = [5, 10, 20]


def gt_frames(scene):
    out = []
    for js in sorted(glob.glob(f"{DS}/label/{scene}/*.json")):
        if js.endswith(".jpg.json"):
            continue
        d = json.load(open(js))
        h, w = d["info"]["height"], d["info"]["width"]
        fr = d["info"]["name"].split(".jpg")[0]
        objs = []
        for o in d["objects"]:
            m = np.zeros((h, w), np.uint8)
            poly = np.asarray(o["segmentation"], np.float32).reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(m, [poly], 1)
            if m.sum() < MIN_PX:
                continue
            objs.append(dict(label=o["category"],
                             bbox=np.asarray(o["bbox"], float).reshape(-1), mask=m))
        if objs:
            out.append((fr, objs))
    return out


def jkernel(mask, pct):
    k = int(round(pct / 100 * np.sqrt(mask.sum())))
    k = max(3, k | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def make_variants(R, obj, rng):
    """返回 [(variant_name, q_normalized)] — 扰动只作用于查询构造."""
    mask, bbox = obj['mask'], obj['bbox']
    q = R[mask > 0].mean(axis=0)
    q = q / (np.linalg.norm(q) + 1e-8)
    out = [('base', q)]
    k = jkernel(mask, 10)
    for pct in JITTER_PCTS:
        kd = jkernel(mask, pct)
        for tag, op in (('dil', cv2.dilate), ('ero', cv2.erode)):
            m2 = op(mask, kd)
            if m2.sum() >= MIN_PX:
                qv = R[m2 > 0].mean(axis=0)
                out.append((f'{tag}{pct}', qv / (np.linalg.norm(qv) + 1e-8)))
    x1, y1, x2, y2 = bbox.astype(int)
    mb = np.zeros_like(mask)
    mb[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = 1
    if mb.sum() >= MIN_PX:
        qv = R[mb > 0].mean(axis=0)
        out.append(('bboxrect', qv / (np.linalg.norm(qv) + 1e-8)))
    for s in NOISE_SIGMAS:
        qv = q + s * rng.standard_normal(q.shape).astype(np.float32)
        out.append((f'noise{s:g}', qv / (np.linalg.norm(qv) + 1e-8)))
    ring = cv2.dilate(mask, jkernel(mask, 25)) & ~mask
    if ring.sum() >= MIN_PX:
        qb = R[ring > 0].mean(axis=0)
        qb = qb / (np.linalg.norm(qb) + 1e-8)
        for a in MIX_ALPHAS:
            qv = (1 - a) * q + a * qb
            out.append((f'mix{a:g}', qv / (np.linalg.norm(qv) + 1e-8)))
    return out


def eval_field(scene, field, gts, rng):
    rdir = os.path.join(OUT, field, 'train', 'ours_None', 'renders_npy')
    frames = [fr for fr, _ in gts
              if os.path.exists(os.path.join(rdir, f'{render_idx(fr):05d}.npy'))]
    gtmap = {fr: objs for fr, objs in gts if fr in frames}
    # pass 1: 每查询帧构造全部扰动变体
    variants = {}   # frA -> list of (obj_idx, label, vname, q)
    for frA in frames:
        R = np.load(os.path.join(rdir, f'{render_idx(frA):05d}.npy')).astype(np.float32)
        variants[frA] = [(i, o['label'], vn, q)
                         for i, o in enumerate(gtmap[frA])
                         for vn, q in make_variants(R, o, rng)]
    names = [vn for _, _, vn, _ in variants[frames[0]]]
    # pass 2: frB 外层, 每帧只归一化一次
    hits = {vn: {} for vn in names}          # vname -> {(frA,i): [hit,n]} 按 (查询帧,物体)
    frame_acc = {vn: {} for vn in names}     # vname -> {frA: [hit,n]}
    for frB in frames:
        R = np.load(os.path.join(rdir, f'{render_idx(frB):05d}.npy')).astype(np.float32)
        Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
        for frA in frames:
            if frA == frB:
                continue
            same = {}
            for o in gtmap[frB]:
                same.setdefault(o['label'], []).append(o['bbox'])
            for i, lab, vn, q in variants[frA]:
                bbs = same.get(lab)
                if not bbs:
                    continue
                simf = cv2.filter2D((Rn @ q).reshape(R.shape[:2]), -1, KERNEL)
                y, x = np.unravel_index(np.argmax(simf), simf.shape)
                h = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bbs)
                hits[vn].setdefault((frA, i), [0, 0])
                hits[vn][(frA, i)][0] += h
                hits[vn][(frA, i)][1] += 1
                frame_acc[vn].setdefault(frA, [0, 0])
                frame_acc[vn][frA][0] += h
                frame_acc[vn][frA][1] += 1
    return hits, frame_acc


def bootstrap_ci(units, reps=1000, seed=0):
    """units: [(hit, n)] 按查询物体单元; 重采样单元, 按 trial 数加权."""
    rng = np.random.default_rng(seed)
    H = np.array([u[0] for u in units], float)
    N = np.array([u[1] for u in units], float)
    idx = rng.integers(0, len(H), (reps, len(H)))
    m = H[idx].sum(axis=1) / N[idx].sum(axis=1)
    return np.percentile(m, [2.5, 97.5]) * 100


def run(scene):
    gts = gt_frames(scene)
    rng = np.random.default_rng(0)
    print(f"\n===== [{scene}] =====")
    for tag, field in FIELDS[scene]:
        hits, frame_acc = eval_field(scene, field, gts, rng)
        tot_b = sum(h for h, _ in hits['base'].values())
        n_b = sum(n for _, n in hits['base'].values())
        lo, hi = bootstrap_ci(list(hits['base'].values()))
        fa = frame_acc['base']
        accs = np.array([100 * h / n for h, n in fa.values()])
        print(f"[{scene}] {tag} ({field})")
        print(f"  (a) base: {tot_b}/{n_b} ({100 * tot_b / max(n_b, 1):.1f}%) "
              f"bootstrap95%CI [{lo:.1f},{hi:.1f}] | per-query-frame acc: "
              f"mean {accs.mean():.1f} std {accs.std():.1f} min {accs.min():.1f} max {accs.max():.1f} "
              f"(n_frames={len(accs)})")
        order = (['base'] + [f'dil{p}' for p in JITTER_PCTS] + [f'ero{p}' for p in JITTER_PCTS]
                 + ['bboxrect'] + [f'noise{s:g}' for s in NOISE_SIGMAS]
                 + [f'mix{a:g}' for a in MIX_ALPHAS])
        for vn in order:
            if vn not in hits or not hits[vn]:
                continue
            tot = sum(h for h, _ in hits[vn].values())
            n = sum(n for _, n in hits[vn].values())
            print(f"  (b) {vn:10s} hit={tot}/{n} ({100 * tot / max(n, 1):.1f}%)")


if __name__ == '__main__':
    for scene in FIELDS:
        run(scene)
