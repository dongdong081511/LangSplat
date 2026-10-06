#!/bin/bash
# EXP-044: training-time EMA cross-view tile consistency regularizer.
# Field trains on RAW 2D supervision (language_features_dim32_dino) + lf_cons loss;
# the offline-smoothing (EXP-043) alternative. Smoke test 300 iters first.
# Usage: LF_CONS_W=0.1 TAG=w01 bash run_exp044.sh [scene]
set -e; set -o pipefail
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
SC=${1:-waldo_kitchen}
W=${LF_CONS_W:-0.1}
M=${LF_CONS_M:-0.9}
TAG=${TAG:-w01}
echo "==== EXP-044 start scene=$SC tag=$TAG w=$W m=$M $(date +%H:%M:%S)"

grep -q "NUM_CHANNELS_language_feature 32" submodules/langsplat-rasterization/cuda_rasterizer/config.h || { echo RASTERIZER_NOT_32D; exit 1; }

render_if_missing() {
  local D=$1
  if [ -d "$D/train/ours_None/renders_npy" ] && [ "$(ls $D/train/ours_None/renders_npy | wc -l)" -gt 100 ]; then
    echo "skip render $D (exists)"; return
  fi
  echo "--- render $D $(date +%H:%M:%S)"
  $PY -u render.py -m $D --include_feature 2>&1 | tail -1
  local N=$(ls $D/train/ours_None/renders_npy 2>/dev/null | wc -l)
  echo "renders: $N"; [ "$N" -gt 100 ] || { echo RENDER_FAIL $D; exit 1; }
}

# smoke: 300-iter run to validate the new loss path (fails fast on code errors)
if [ ! -f output/smoke_044_$SC/.smoke_done ]; then
  $PY -u train.py -s dataset/lerf_ovs/$SC -m output/smoke_044_$SC \
      --language_features_name language_features_dim32_dino --feature_level 1 --include_feature \
      --start_checkpoint dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth \
      --lf_cons_weight $W --lf_cons_momentum $M --iterations 300 --port 6079 2>&1 | tail -3
  mkdir -p output/smoke_044_$SC && touch output/smoke_044_$SC/.smoke_done
  echo "smoke OK"
fi

for L in 1 2 3; do
  [ -f output/${SC}_dino_32dc_${TAG}_$L/chkpnt30000.pth ] && continue
  echo "--- train cons level $L $(date +%H:%M:%S)"
  $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32dc_${TAG} \
      --language_features_name language_features_dim32_dino --feature_level $L --include_feature \
      --start_checkpoint dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth \
      --lf_cons_weight $W --lf_cons_momentum $M --port 607$L 2>&1 | tail -2
  ls output/${SC}_dino_32dc_${TAG}_$L/chkpnt30000.pth || { echo TRAIN_FAIL L$L; exit 1; }
done
for L in 1 2 3; do render_if_missing output/${SC}_dino_32dc_${TAG}_$L; done

cd eval
echo "--- EVAL ${SC}_DINO_c${TAG} $(date +%H:%M:%S)"
ABBR=$(echo $SC | cut -c1-4 | tr 'a-z' 'A-Z')
case $SC in
  waldo_kitchen) ABBR=WALDO;;
  teatime) ABBR=TEATIME;;
  figurines) ABBR=FIGURINES; CLIPJ=FIGURINES_CLIP24d.json;;
  ramen) ABBR=RAMEN;;
esac
CLIPJ=${CLIPJ:-${ABBR}_CLIP8d.json}
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC \
  --scene_root ../dataset/lerf_ovs/$SC \
  --db_render_dirs ../output/${SC}_dino_32dc_${TAG}_1 ../output/${SC}_dino_32dc_${TAG}_2 ../output/${SC}_dino_32dc_${TAG}_3 \
  --query_subdir language_features_dim32_dino --tag ${ABBR}_DINO_c${TAG} \
  --out_json ../eval_result/mcnemar/${ABBR}_DINO_c${TAG}.json 2>&1 | grep -E "^\["
for P in first any; do
  $PY -u mcnemar.py --clip_json ../eval_result/mcnemar/$CLIPJ \
    --dino_json ../eval_result/mcnemar/${ABBR}_DINO_c${TAG}.json --protocol $P \
    --tag ${ABBR}_c${TAG}_vs_CLIP_$P
done
cd ..
echo "==== EXP-044 ALL DONE $(date +%H:%M:%S)"
