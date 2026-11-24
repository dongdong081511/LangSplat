"""McNemar exact test on paired image-query hits between two 3D fields (EXP-041).

Reads two per-pair JSONs produced by eval_image_query.py --out_json, joins on
(frameA, obj_i), and tests whether the two fields' hit/miss vectors differ
significantly (exact binomial on discordant pairs).
"""
import json
import argparse
from math import comb


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d, {(x['frameA'], x['obj_i']): x for x in d['pairs']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_json', required=True)
    ap.add_argument('--dino_json', required=True)
    ap.add_argument('--protocol', default='any', choices=['first', 'any'])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    dc, C = load(a.clip_json)
    dd, D = load(a.dino_json)
    common = sorted(set(C) & set(D))
    hk = 'hit_' + a.protocol
    n = len(common)
    if n == 0:
        print(f'[{a.tag}] ERROR: no common pairs'); return

    b = sum(1 for k in common if C[k][hk] and not D[k][hk])  # clip-only hits
    c = sum(1 for k in common if not C[k][hk] and D[k][hk])  # dino-only hits
    m = b + c
    # two-sided exact binomial test on discordant pairs
    if m == 0:
        p = 1.0
    else:
        tail = sum(comb(m, k) for k in range(0, min(b, c) + 1)) / 2 ** m
        p = min(1.0, 2 * tail)

    ch = sum(C[k][hk] for k in common)
    dh = sum(D[k][hk] for k in common)
    print(f'[{a.tag}] protocol={a.protocol} n_pairs={n} '
          f'CLIP {ch}/{n}={ch / n:.2%}  DINO {dh}/{n}={dh / n:.2%}  '
          f'delta={100 * (dh - ch) / n:+.2f}pp | '
          f'discordant: b(clip-only)={b} c(dino-only)={c} | '
          f'McNemar exact p={p:.4f} {"*" if p < 0.05 else "ns"}')


if __name__ == '__main__':
    main()
