#!/bin/bash
# EXP-045c: figurines lf_cons weight grid extension (w in {0.5, 0.7})
# Waits for run_exp045b_grid.sh ({0.15, 0.2}) to finish, then runs automatically.
# Same tag rule as 045b: json = FIGURINES_DINO_ccw<NN>.json (0.5->05, 0.7->07),
# renders freed after each config's eval (keep ckpt).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/eval"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SC=figurines

# wait for the running 045b grid to finish (max ~6h guard)
for i in $(seq 1 72); do
  pgrep -f "run_exp045b_grid.sh" > /dev/null || break
  sleep 300
done
pgrep -f "run_exp045b_grid.sh" > /dev/null && { echo "045b still running after 6h, abort"; exit 1; }
echo "=== 045b finished, starting extension $(date +%H:%M:%S)"

for W in 0.5 0.7; do
  TAG=$(echo $W | tr -d '.')            # 05 / 07
  echo "=== figurines w=$W tag=cw$TAG $(date +%H:%M:%S)"

  (cd "$ROOT" && LF_CONS_W=$W LF_CONS_M=0.9 TAG=cw$TAG bash run_exp044.sh $SC)

  for L in 1 2 3; do
    D=../output/${SC}_dino_32dc_cw${TAG}_${L}/train/ours_None
    rm -rf "$D/renders_npy" "$D/renders"
  done
  echo "renders freed for cw$TAG $(df -h / | tail -1 | awk '{print $4}') free"

  for P in first any; do
    $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/FIGURINES_DINO_a07.json \
      --dino_json ../eval_result/mcnemar/FIGURINES_DINO_ccw$TAG.json --protocol $P \
      --tag FIGURINES_ccw$TAG_vs_a07_$P || echo "vs_a07 skip (cw$TAG)"
  done
done

echo "EXP-045c figurines grid extension ALL DONE $(date +%H:%M:%S)"
