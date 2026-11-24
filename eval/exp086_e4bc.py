# exp086_e4bc.py — E4(b)(c) 自蒸馏机制直接度量 (离线, 依赖 run_exp085_e4a.sh 产物)
#
# (b) top-1 命中位置像素偏移分布: 穷举 GT 帧对 RM 协议 (query=帧A GT mask 内渲染均值,
#     DB=帧B 渲染余弦图→30×30 滤波→argmax), 记录峰值到最近同名 GT bbox 中心的像素距离
#     (绝对 px 与 bbox 对角线归一化), 按 hit/miss 分层 → 验证 "smoothing repairs
#     bbox-center offsets, not discriminability": blend 后 miss 应从远漂移变为近偏移.
# (c) 跨视角方差 vs α: 每物体渲染特征跨视角成对余弦 (E1 一致性度量) 随 α 的变化
#     → 直接支撑 "pure self-distillation overfits single frames" (α=raw 最低).
# 用法: python eval/exp086_e4bc.py 2>&1 | tee eval_result/exp086_e4bc.log
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

# (场景, GT目录) 与 α→场 base; a05/a09 由 EXP-085 产出
CONFIGS = [
    ('ramen', [
        ('raw', 'ramen_dino_32d'),
        ('a05', 'ramen_dino_32ds_a05'),
        ('a07', 'ramen_dino_32ds_a07'),
        ('a09', 'ramen_dino_32ds_a09'),
    ]),
    ('waldo_kitchen', [
        ('raw', 'waldo_kitchen_dino_32d'),
        ('a05', 'waldo_kitchen_dino_32ds_a05'),
        ('a07', 'waldo_kitchen_dino_32ds_a07'),
        ('a09', 'waldo_kitchen_dino_32ds_a09'),
    ]),
    ('teatime', [
        ('raw', 'teatime_dino_32d'),
        ('a03', 'teatime_dino_32ds_a03'),
        ('a05', 'teatime_dino_32ds_a05'),
        ('a07', 'teatime_dino_32ds_a07'),
        ('a09', 'teatime_dino_32ds_a09'),
    ]),
    ('figurines', [
        ('raw', 'figurines_dino_32d'),
        ('a05', 'figurines_dino_32ds_a05'),
        ('a07', 'figurines_dino32ds_a07'),
        ('a09', 'figurines_dino_32ds_a09'),
    ]),
]

render_idx = lambda name: (int(name[6:].split('.')[0]) - 1 if name.startswith('frame_')
                           else int(name.split('.')[0]))


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


def analyze(scene, tag, field, gts):
    rdir = os.path.join(OUT, field + '_1', 'train', 'ours_None', 'renders_npy')
    frames = [fr for fr, _ in gts
              if os.path.exists(os.path.join(rdir, f'{render_idx(fr):05d}.npy'))]
    gtmap = {fr: objs for fr, objs in gts if fr in frames}
    # pass 1: queries + 3D view features (一致性用)
    queries, views = [], {}
    for frA in frames:
        R = np.load(os.path.join(rdir, f'{render_idx(frA):05d}.npy')).astype(np.float32)
        for i, o in enumerate(gtmap[frA]):
            q = R[o['mask'] > 0].mean(axis=0)
            q = q / (np.linalg.norm(q) + 1e-8)
            queries.append((frA, i, o['label'], q))
            views.setdefault(o['label'], {})[frA] = q
    # (c) 跨视角一致性
    cons = []
    for lab, fs in views.items():
        if len(fs) < 2:
            continue
        V = np.stack(list(fs.values()))
        iu = np.triu_indices(len(fs), 1)
        cons.append(float((V @ V.T)[iu].mean()))
    # pass 2: 检索 + 偏移
    dists_all, dists_miss, ndists_miss, hits, n = [], [], [], 0, 0
    from collections import defaultdict
    by_frame = defaultdict(list)
    for frA, i, lab, q in queries:
        by_frame[frA].append((i, lab, q))
    for frB in frames:
        R = np.load(os.path.join(rdir, f'{render_idx(frB):05d}.npy')).astype(np.float32)
        Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
        for frA in frames:
            if frA == frB:
                continue
            for i, lab, q in by_frame[frA]:
                bbs = [o['bbox'] for o in gtmap[frB] if o['label'] == lab]
                if not bbs:
                    continue
                simf = cv2.filter2D((Rn @ q).reshape(R.shape[:2]), -1, KERNEL)
                y, x = np.unravel_index(np.argmax(simf), simf.shape)
                d2 = [np.hypot(x - (b[0] + b[2]) / 2, y - (b[1] + b[3]) / 2) for b in bbs]
                j = int(np.argmin(d2))
                dg = d2[j]
                diag = np.hypot(bbs[j][2] - bbs[j][0], bbs[j][3] - bbs[j][1]) + 1e-9
                inside = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bbs)
                n += 1
                hits += inside
                dists_all.append(dg)
                if not inside:
                    dists_miss.append(dg)
                    ndists_miss.append(dg / diag)
    a = 100 * hits / max(n, 1)
    med = float(np.median(dists_all)) if dists_all else float('nan')
    mmed = float(np.median(dists_miss)) if dists_miss else float('nan')
    mp90 = float(np.percentile(dists_miss, 90)) if dists_miss else float('nan')
    mnmed = float(np.median(ndists_miss)) if ndists_miss else float('nan')
    c = float(np.mean(cons)) if cons else float('nan')
    print(f"[{scene}] α={tag:4s} acc={a:5.1f}% ({hits}/{n}) | peak→bbox-center px: "
          f"median_all {med:6.1f} | miss-only median {mmed:6.1f} p90 {mp90:6.1f} "
          f"(norm {mnmed:.2f}×diag) | cross-view cos {c:.4f}")
    return dict(scene=scene, alpha=tag, acc=a, n=n, med_all=med, med_miss=mmed,
                p90_miss=mp90, med_miss_norm=mnmed, cons=c)


if __name__ == '__main__':
    rows = []
    for scene, fields in CONFIGS:
        gts = gt_frames(scene)
        for tag, field in fields:
            try:
                rows.append(analyze(scene, tag, field, gts))
            except Exception as e:
                print(f"[{scene}] α={tag} ERROR: {e}")
    # α 曲线汇总
    print('\n=== (c) cross-view consistency & accuracy vs α ===')
    for scene in dict.fromkeys(r['scene'] for r in rows):
        rs = [r for r in rows if r['scene'] == scene]
        print(f"{scene:14s} " + " | ".join(
            f"α={r['alpha']}: acc {r['acc']:.1f}%, cos {r['cons']:.4f}" for r in rs))
