#!/bin/bash
# EXP-045b: figurines lf_cons weight grid (fill 0.1~0.3 window: w in {0.15, 0.2})
# Reuses run_exp044.sh (tag rule: eval tag = ${ABBR}_DINO_c${TAG}, we pass TAG=cw<NN>
#   -> json = FIGURINES_DINO_ccw<NN>.json, model dirs = figurines_dino_32dc_cw<NN>_<L>)
# Disk policy: after each config's eval, delete renders_npy/renders (keep ckpt, re-renderable).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/eval"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SC=figurines

for W in 0.15 0.2; do
  TAG=$(echo $W | tr -d '.')            # 015 / 02
  echo "=== figurines w=$W tag=cw$TAG $(date +%H:%M:%S)"

  (cd "$ROOT" && LF_CONS_W=$W LF_CONS_M=0.9 TAG=cw$TAG bash run_exp044.sh $SC)

  # free renders (keep ckpt) to fit the next config on disk
  for L in 1 2 3; do
    D=../output/${SC}_dino_32dc_cw${TAG}_${L}/train/ours_None
    rm -rf "$D/renders_npy" "$D/renders"
  done
  echo "renders freed for cw$TAG $(df -h / | tail -1 | awk '{print $4}') free"

  # reference McNemar: vs a07 field (CLIP24d 对照已在 run_exp044 内跑过)
  for P in first any; do
    $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/FIGURINES_DINO_a07.json \
      --dino_json ../eval_result/mcnemar/FIGURINES_DINO_ccw$TAG.json --protocol $P \
      --tag FIGURINES_ccw$TAG_vs_a07_$P || echo "vs_a07 skip (cw$TAG)"
  done
done

echo "EXP-045b figurines grid ALL DONE $(date +%H:%M:%S)"
