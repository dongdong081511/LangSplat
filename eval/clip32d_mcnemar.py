# P0-3: CLIP-32d vs DINO-32d 配对 McNemar (majority 口径, 同 query 集)。
# b = DINO-32d hit & CLIP-32d miss, c = CLIP-32d hit & DINO-32d miss;
# 两尾 exact binomial。用法: python eval/clip32d_mcnemar.py
import json, os
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC = os.path.join(ROOT, 'eval_result', 'mcnemar')

PAIRS = {  # scene -> (CLIP32d json, DINO a07 maj json)
    'teatime':       ('teatime_CLIP32d.json', 'teatime_DINO_a07_maj.json'),
    'figurines':     ('figurines_CLIP32d.json', 'figurines_DINO_a07_maj.json'),
    'ramen':         ('ramen_CLIP32d.json', 'ramen_DINO_a07_maj.json'),
    'waldo_kitchen': ('waldo_kitchen_CLIP32d.json', 'waldo_kitchen_DINO_a07_maj.json'),
}

rows, B, C = [], 0, 0
for sc, (f32, fdino) in PAIRS.items():
    a = {(p['frameA'], p['obj_i']): p['mf_majority']
         for p in json.load(open(os.path.join(MC, f32)))['pairs']}
    d = {(p['frameA'], p['obj_i']): p['mf_majority']
         for p in json.load(open(os.path.join(MC, fdino)))['pairs']}
    keys = sorted(set(a) & set(d))
    assert len(keys) == len(a) == len(d), f'{sc}: key mismatch {len(a)}/{len(d)}/{len(keys)}'
    b = sum(1 for k in keys if d[k] and not a[k])
    c = sum(1 for k in keys if a[k] and not d[k])
    B += b; C += c
    pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    rows.append((sc, len(keys), sum(a.values()), sum(d.values()), b, c, pv))
    print(f'{sc:15s} n={len(keys)} CLIP32d {sum(a.values())} DINO32d {sum(d.values())} '
          f'b:c={b}:{c}  p={pv:.4g}')

n = B + C
pv = binomtest(B, n, 0.5).pvalue if n else 1.0
print(f'POOLED          b:c={B}:{C}  p={pv:.4g}')
json.dump(dict(protocol='majority, paired exact binomial McNemar; b=DINO32d win, c=CLIP32d win',
               rows=[dict(scene=r[0], n=r[1], clip32d=r[2], dino32d=r[3], b=r[4], c=r[5],
                          p_value=r[6]) for r in rows],
               pooled=dict(b=B, c=C, p_value=pv)),
          open(os.path.join(MC, 'CLIP32d_vs_DINO32d_mcnemar.json'), 'w'), indent=1)
print('saved -> eval_result/mcnemar/CLIP32d_vs_DINO32d_mcnemar.json')
