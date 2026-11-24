# P1-1 (CVPR2027 审稿): LUDVIG 外部 baseline 受限匹配 + paired McNemar。
# 受限匹配规则: 按 (query_frame, label) 把 LUDVIG per-frame 命中受限到我们论文协议的
#   目标帧集合 (GT 中存在同 label 的全部帧, 除源帧; 见 eval/eval_image_query.py 主循环),
#   在受限集合上重算 LUDVIG majority (mean>=0.5), 与我们路由场的 mf_majority 配对,
#   算不一致对 b (Ours hit & LUDVIG miss) / c (LUDVIG hit & Ours miss) +
#   两尾 exact binomial McNemar。
# 用法 (langsplat env):
#   python eval/ludvig_restricted_mcnemar.py
import json, glob, os
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_BASE = os.path.join(ROOT, 'dataset', 'lerf_ovs', 'label')
LUDVIG_DIR = os.path.join(ROOT, '..', 'ludvig', 'eval_result', 'ludvig')
OUT_JSON = os.path.join(ROOT, 'eval_result', 'mcnemar', 'LUDVIG_restricted.json')

# 论文路由场 (Tab. routing_main majority): teatime/figurines/ramen -> DINO a07, waldo -> CLIP 8d
ROUTED = {
    'teatime':       'teatime_DINO_a07_maj.json',
    'figurines':     'figurines_DINO_a07_maj.json',
    'ramen':         'ramen_DINO_a07_maj.json',
    'waldo_kitchen': 'waldo_kitchen_CLIP8d_maj.json',
}


def load_gt_labels(gt_dir):
    """frame -> set(categories); 与 eval_image_query.load_gt 的帧名/label 口径一致."""
    gt = {}
    for js in sorted(glob.glob(os.path.join(gt_dir, '*.json'))):
        with open(js) as f:
            d = json.load(f)
        frame = d['info']['name'].split('.jpg')[0]
        gt[frame] = {o['category'] for o in d['objects']}
    return gt


def target_frames(gt, frA, label):
    return [fr for fr in sorted(gt) if fr != frA and label in gt[fr]]


rows, pooled_b, pooled_c = [], 0, 0
for sc, ours_file in ROUTED.items():
    gt = load_gt_labels(os.path.join(GT_BASE, sc))
    ours = json.load(open(os.path.join(ROOT, 'eval_result', 'mcnemar', ours_file)))
    lud = json.load(open(os.path.join(LUDVIG_DIR, f'{sc}_LUDVIG.json')))
    lud_map = {(p['query_frame'], p['label']): p['per_frame'] for p in lud['pairs']}

    b = c = n_matched = n_missing_frames = 0
    ours_hits = lud_hits = 0
    for p in ours['pairs']:
        key = (p['frameA'], p['label'])
        if key not in lud_map:
            continue
        pf = lud_map[key]
        tf = target_frames(gt, p['frameA'], p['label'])
        restricted = [pf[fr] for fr in tf if fr in pf]
        n_missing_frames += len(tf) - len(restricted)
        lud_maj = int(np.mean(restricted) >= 0.5)
        ours_maj = int(p['mf_majority'])
        n_matched += 1
        ours_hits += ours_maj
        lud_hits += lud_maj
        if ours_maj and not lud_maj:
            b += 1
        elif lud_maj and not ours_maj:
            c += 1
    pooled_b += b
    pooled_c += c
    n = b + c
    pv = binomtest(b, n, 0.5).pvalue if n > 0 else 1.0
    rows.append(dict(scene=sc, n_ours=len(ours['pairs']), n_ludvig=lud['n_pairs'],
                     n_matched=n_matched, ours_hits=ours_hits, ludvig_hits=lud_hits,
                     b_ours_win=b, c_ludvig_win=c, p_value=round(pv, 6),
                     missing_target_frames=n_missing_frames))
    print(f"{sc:15s} matched {n_matched:3d}  Ours {ours_hits}/{n_matched}  "
          f"LUDVIG {lud_hits}/{n_matched}  b:c={b}:{c}  p={pv:.4g}  "
          f"(LUDVIG 缺失目标帧 {n_missing_frames})")

n = pooled_b + pooled_c
pv = binomtest(pooled_b, n, 0.5).pvalue if n > 0 else 1.0
rows.append(dict(scene='pooled', b_ours_win=pooled_b, c_ludvig_win=pooled_c,
                 p_value=pv))
print(f"POOLED          b:c={pooled_b}:{pooled_c}  p={pv:.4g}")

json.dump(dict(protocol='restricted matching to paper target frames, majority>=0.5, '
                       'two-sided exact binomial McNemar; b=Ours hit & LUDVIG miss, '
                       'c=LUDVIG hit & Ours miss', rows=rows),
          open(OUT_JSON, 'w'), indent=1)
print('saved ->', OUT_JSON)
