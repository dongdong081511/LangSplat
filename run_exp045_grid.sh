#!/bin/bash
# EXP-045: grid search lf_cons weight on teatime (w in {0.3, 1.0, 3.0}; w=0.1 done in EXP-044 = raw).
# Each config: 3-level train + render + eval + McNemar vs CLIP (in run_exp044.sh), then McNemar vs raw here.
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
echo "==== EXP-045 grid start $(date +%H:%M:%S)"
for W in 0.3 1.0 3.0; do
  TAG=$(echo $W | tr -d '.')
  echo "=== w=$W tag=cw$TAG $(date +%H:%M:%S)"
  LF_CONS_W=$W LF_CONS_M=0.9 TAG=cw$TAG bash run_exp044.sh teatime
  cd eval
  for P in first any; do
    $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/TEATIME_DINO_raw.json \
      --dino_json ../eval_result/mcnemar/TEATIME_DINO_ccw$TAG.json --protocol $P \
      --tag TEATIME_ccw$TAG_vs_raw_$P
  done
  cd ..
done
echo "==== EXP-045 grid ALL DONE $(date +%H:%M:%S)"
