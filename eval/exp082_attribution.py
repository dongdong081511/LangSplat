# exp082_attribution.py — E1 跨视角优势归因 (审稿 D 维度: 2D教师 / 3D判别 / 2D→3D lift / 跨视角一致性)
#
# (a) 2D 教师直接检索: query = 帧A GT mask 内 SAM 段特征重叠加权均值; DB = 帧B 逐段特征;
#     逐像素段分数图 → 30×30 均值滤波 → argmax; hit = 峰值落同名 GT bbox (与 RM 协议逐位同构).
# (b) 3D 场检索: 同协议, query/DB 均来自 renders_npy (即 eval_image_query --query_from_render 的
#     穷举 GT 帧对版, 与 rm_query_compare.py 一致).
# (c) 跨视角一致性: 每物体取全部可见帧 (≥2), 2D 原生段特征 vs 3D 渲染特征的跨视角成对余弦相似度;
#     3D 更高 = 场拟合压缩了视角方差 → 解释 2D→3D lift.
#
# 场景覆盖 (2D 特征存活性决定): DINO 教师 teatime/figurines/ramen/waldo_kitchen,
# CLIP 教师 sofa (28 帧). renders 决定 3D 侧: teatime/ramen/waldo (dino32d), sofa (clip8d).
# 用法: python eval/exp082_attribution.py 2>&1 | tee eval_result/exp082_attribution.log  (纯 CPU)
import os
import sys
import glob
import json
import numpy as np
import cv2

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DS = os.path.join(BASE, 'dataset', 'lerf_ovs')
OUT = os.path.join(BASE, 'output')
KERNEL = np.ones((30, 30)) / 900
MIN_PX = 30

SCENES = [
    dict(scene='teatime',      feat='dino', field='teatime_dino_32ds_a07_1'),
    dict(scene='figurines',    feat='dino', field=None),
    dict(scene='ramen',        feat='dino', field='ramen_dino_32ds_a07_1'),
    dict(scene='waldo_kitchen',feat='dino', field='waldo_kitchen_dino_32ds_a07_1'),
    dict(scene='sofa',         feat='clip', field='sofa_clip8d_1'),
]


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


def feat_dir(scene, feat):
    return os.path.join(DS, scene,
                        'language_features_dino' if feat == 'dino' else 'language_features')


def render_idx(name):
    # LERF: frame_00002 (1-based) → renders 00001; OVS: 02 → renders 00002
    if name.startswith('frame_'):
        return int(name[6:].split('.')[0]) - 1
    return int(name.split('.')[0])


def seg_data(scene, feat, fr):
    sub = feat_dir(scene, feat)
    F = np.load(os.path.join(sub, fr + '_f.npy')).astype(np.float32)
    S = np.load(os.path.join(sub, fr + '_s.npy')).astype(np.int64)[-1]
    return F, S


def seg_query(F, S, mask):
    """GT mask 内段特征重叠加权均值 (等价 mask 内逐像素段特征的像素均值)."""
    ids, cnts = np.unique(S[mask > 0], return_counts=True)
    ok = ids >= 0
    ids, cnts = ids[ok], cnts[ok]
    if len(ids) == 0:
        return None
    q = (cnts[:, None] * F[ids]).sum(axis=0) / cnts.sum()
    return q / (np.linalg.norm(q) + 1e-8)


def render_query(R, mask):
    q = R[mask > 0].mean(axis=0)
    return q / (np.linalg.norm(q) + 1e-8)


def run_scene(cfg):
    scene, feat, field = cfg['scene'], cfg['feat'], cfg['field']
    gts = gt_frames(scene)
    fdir = feat_dir(scene, feat)
    # 只保留 2D 特征存活的帧; 3D 侧再要求 renders 存活
    gts = [(fr, objs) for fr, objs in gts
           if os.path.exists(os.path.join(fdir, fr + '_f.npy'))]
    has3d = []
    if field:
        rdir = os.path.join(OUT, field, 'train', 'ours_None', 'renders_npy')
        has3d = [(fr, objs) for fr, objs in gts
                 if os.path.exists(os.path.join(rdir, f'{render_idx(fr):05d}.npy'))]
    print(f"\n===== [{scene}] feat={feat} field={field} | gt_frames={len(gts)} "
          f"(3d-usable {len(has3d)}) =====")

    queries2d, queries3d = [], []   # (frA, obj_idx, label, q)
    views = {}                      # label -> {fr: (q2d, q3d or None)}
    for fr, objs in gts:
        F, S = seg_data(scene, feat, fr)
        for i, obj in enumerate(objs):
            q = seg_query(F, S, obj['mask'])
            if q is None:
                continue
            queries2d.append((fr, i, obj['label'], q))
            views.setdefault(obj['label'], {})[fr] = [q, None]
    if field:
        rdir = os.path.join(OUT, field, 'train', 'ours_None', 'renders_npy')
        for fr, objs in has3d:
            R = np.load(os.path.join(rdir, f'{render_idx(fr):05d}.npy')).astype(np.float32)
            for i, obj in enumerate(objs):
                q = render_query(R, obj['mask'])
                queries3d.append((fr, i, obj['label'], q))
                if fr in views.get(obj['label'], {}):
                    views[obj['label']][fr][1] = q

    def eval_side(qs, side, frames):
        """qs: (frA, i, label, q); DB 帧外层循环, 每帧只加载一次."""
        from collections import defaultdict
        by_frame = defaultdict(list)
        for frA, i, lab, q in qs:
            by_frame[frA].append((i, lab, q))
        gt_by_frame = dict(frames)
        hit = n = 0
        per_obj = defaultdict(lambda: [0, 0])
        for frB in sorted(by_frame):
            if side == '2d':
                F, S = seg_data(scene, feat, frB)
                Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
                valid = S >= 0
                base = np.where(valid, np.maximum(S, 0), 0)
            else:
                rdir = os.path.join(OUT, field, 'train', 'ours_None', 'renders_npy')
                R = np.load(os.path.join(rdir, f'{render_idx(frB):05d}.npy')).astype(np.float32)
                Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-8)
            gtB = gt_by_frame[frB]
            for frA, rows in by_frame.items():
                if frA == frB:
                    continue
                for i, lab, q in rows:
                    bbs = [o['bbox'] for o in gtB if o['label'] == lab]
                    if not bbs:
                        continue
                    if side == '2d':
                        sc = Fn @ q
                        smap = sc[base].reshape(S.shape)
                        smap[~valid] = -1.0
                    else:
                        smap = (Rn @ q).reshape(R.shape[:2])
                    simf = cv2.filter2D(smap, -1, KERNEL)
                    y, x = np.unravel_index(np.argmax(simf), simf.shape)
                    h = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in bbs)
                    n += 1
                    hit += h
                    per_obj[(frA, i, lab)][0] += h
                    per_obj[(frA, i, lab)][1] += 1
        return hit, n, per_obj

    h2, n2, obj2 = eval_side(queries2d, '2d', gts)
    line = f"[{scene}] 2D teacher ({feat})  hit={h2}/{n2}"
    line += f" ({100 * h2 / max(n2, 1):.1f}%)"
    if field and queries3d:
        h3, n3, obj3 = eval_side(queries3d, '3d', has3d)
        line += f" | 3D field hit={h3}/{n3} ({100 * h3 / max(n3, 1):.1f}%)"
        a2, a3 = 100 * h2 / max(n2, 1), 100 * h3 / max(n3, 1)
        line += f" | lift {a3 - a2:+.1f}pp"
        if len(gts) == len(has3d):
            from collections import defaultdict
            agg2, agg3 = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
            for (frA, i, lab), (ho, no) in obj2.items():
                agg2[lab][0] += ho; agg2[lab][1] += no
            for (frA, i, lab), (ho, no) in obj3.items():
                agg3[lab][0] += ho; agg3[lab][1] += no
            print(f"    {'label':24s} {'2D teacher':>12s} {'3D field':>12s}")
            for lab in sorted(agg2):
                h2o, n2o = agg2[lab]
                h3o, n3o = agg3.get(lab, [0, 0])
                print(f"    {lab:24s} {h2o:>5d}/{n2o:<6d} {h3o:>5d}/{n3o:<6d}")
    print(line)

    # 跨视角一致性 (每物体 ≥2 可见帧)
    if field:
        rows = []
        for lab, fs in sorted(views.items()):
            v2 = [v[0] for v in fs.values() if v[0] is not None]
            v3 = [v[1] for v in fs.values() if v[1] is not None]
            if len(v2) < 2 or len(v3) < 2:
                continue
            V2 = np.stack(v2)
            V3 = np.stack(v3)
            c2 = (V2 @ V2.T)[np.triu_indices(len(v2), 1)].mean()
            c3 = (V3 @ V3.T)[np.triu_indices(len(v3), 1)].mean()
            rows.append((lab, len(v2), c2, c3))
        if rows:
            m2 = float(np.mean([r[2] for r in rows]))
            m3 = float(np.mean([r[3] for r in rows]))
            print(f"[{scene}] cross-view consistency (objects={len(rows)}): "
                  f"2D mean-pairwise-cos {m2:.4f} | 3D {m3:.4f} | Δ {m3 - m2:+.4f} "
                  f"| var reduction {100 * (1 - (1 - m3) / max(1 - m2, 1e-9)):.1f}%")
            for lab, k, c2, c3 in rows:
                print(f"    {lab:24s} views={k:3d}  2D {c2:.4f}  3D {c3:.4f}  Δ {c3 - c2:+.4f}")


if __name__ == '__main__':
    for cfg in SCENES:
        try:
            run_scene(cfg)
        except Exception as e:
            print(f"[{cfg['scene']}] ERROR: {e}")
            import traceback
            traceback.print_exc()
