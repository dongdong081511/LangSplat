#!/bin/bash
# Pack the IQ-Retrieve3D benchmark release: GT labels + eval code + per-pair results.
# Dataset images/renders are NOT included (LERF license); reproduce via PROTOCOL.md Sec.6.
set -e
cd /home/xiedexia/project/LangSplat
R=benchmark/release
mkdir -p $R/gt $R/results $R/code
cp benchmark/PROTOCOL.md $R/
cp eval/eval_image_query.py eval/mcnemar.py eval/smooth_tiles.py eval/smooth_tiles_adaptive.py $R/code/
for SC in teatime figurines waldo_kitchen ramen; do
  mkdir -p $R/gt/$SC
  cp dataset/lerf_ovs/label/$SC/frame_*.json $R/gt/$SC/ 2>/dev/null || echo "no GT for $SC"
done
cp eval_result/mcnemar/*.json $R/results/ 2>/dev/null || true
echo "packed -> $R"
du -sh $R
