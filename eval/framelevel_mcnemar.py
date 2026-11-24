# C2 (CVPR 审稿回应): 帧级 pooled McNemar——把配对单位从 query 降到 (query, frame)。
# 回答审稿质疑 "router 只在 majority 协议下显著, 单帧配对与抛硬币无异":
# 在帧粒度上直接配对检验 CLIP per-scene dims 场 vs DINO a07 场。
# 两个口径:
#   any    = 帧命中 = 任一 level top1 命中 (与论文 majority 聚合的 level 语义一致)
#   chosen = 帧命中 = 该 query 的 chosen-level top1 命中 (部署口径: 单场单 level)
# 数据: eval_result/framelevel/{SC}_{CLIP,DINO}.json (run_c2_framelevel.sh 重采, 含 frame_hits_lv)。
# 用法: python eval/framelevel_mcnemar.py
import json, os
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'eval_result', 'framelevel')
SCENES = ['teatime', 'figurines', 'ramen', 'waldo_kitchen']
CLIP_FILE = {'teatime': 'teatime_CLIP8d_maj', 'figurines': 'figurines_CLIP24d_maj',
             'ramen': 'ramen_CLIP8d_maj', 'waldo_kitchen': 'waldo_kitchen_CLIP8d_maj'}

def load(sc, field):
    d = json.load(open(os.path.join(SRC, f'{sc}_{field}.json')))
    out = {}
    for q in d['pairs']:
        lv = q['frame_hits_lv']
        frames = [f for f, _ in lv['1']]  # 帧序 (各 level 帧集相同)
        anyv = [max(h) for h in zip(*[[h for _, h in lv[k]] for k in sorted(lv)])]
        chosen = [h for _, h in lv[str(q['level'])]]
        assert len(anyv) == len(chosen) == len(frames)
        out[(q['frameA'], q['obj_i'])] = dict(zip(frames, zip(anyv, chosen)))
    return out

rows, B, C = [], 0, 0
for mode in ['any', 'chosen']:
    Bi = Ci = 0
    for sc in SCENES:
        c = load(sc, 'CLIP'); d = load(sc, 'DINO')
        assert set(c) == set(d)
        b = cwin = 0
        for k in c:
            assert set(c[k]) == set(d[k]), f'frame set mismatch {sc} {k}'
            for fr in c[k]:
                dh, ch = d[k][fr][mode == 'chosen'], c[k][fr][mode == 'chosen']
                b += int(dh and not ch); cwin += int(ch and not dh)
        n = b + cwin
        pv = binomtest(b, n, 0.5).pvalue if n else 1.0
        Bi += b; Ci += cwin
        rows.append((mode, sc, b, cwin, pv))
        print(f'{mode:6s} {sc:15s} b:c = {b}:{cwin} (DINO:CLIP)  p = {pv:.4g}')
    pv = binomtest(Bi, Bi + Ci, 0.5).pvalue if Bi + Ci else 1.0
    rows.append((mode, 'pooled', Bi, Ci, pv))
    print(f'{mode:6s} POOLED          b:c = {Bi}:{Ci}  p = {pv:.4g}')
    B, C = Bi, Ci

json.dump(dict(protocol='frame-level paired exact binomial McNemar; b=DINO win, c=CLIP win; '
               'any=any-level per frame, chosen=chosen-level per frame',
               rows=[dict(mode=r[0], scene=r[1], b=r[2], c=r[3], p_value=r[4]) for r in rows]),
          open(os.path.join(SRC, 'framelevel_mcnemar.json'), 'w'), indent=1)
print('saved -> eval_result/framelevel/framelevel_mcnemar.json')
