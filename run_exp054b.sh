#!/bin/bash
# EXP-054b: ramen alpha=0.9 phase (path bugs in run_exp054.sh fixed: mcnemar called from project root)
set -e
cd "$(dirname "$0")"
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"

SC=ramen
CKPT=dataset/lerf_ovs/$SC/output/${SC}_-1/chkpnt30000.pth
$PY eval/smooth_tiles.py --scene $SC --alpha 0.9 --src_subdir language_features_dim32_dino \
  --render_base ${SC}_dino_32ds_a07 --iter_tag a09 2>&1 | tail -1
echo "=== ramen a09 train start $(date +%H:%M:%S)"
for L in 1 2 3; do
  $PY -u train.py -s dataset/lerf_ovs/$SC -m output/${SC}_dino_32ds_a09 \
    --language_features_name language_features_dim32_dino_smooth_a09 --feature_level $L \
    --include_feature --start_checkpoint $CKPT --port 6172 2>&1 | tail -1
  ls output/${SC}_dino_32ds_a09_$L/chkpnt30000.pth || { echo TRAIN_FAIL ramen L$L; exit 1; }
  $PY -u render.py -m output/${SC}_dino_32ds_a09_$L --include_feature 2>&1 | tail -1
  ls output/${SC}_dino_32ds_a09_$L/train/ours_None/renders_npy/00000.npy || { echo RENDER_FAIL ramen L$L; exit 1; }
done
cd eval
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SC --scene_root ../dataset/lerf_ovs/$SC \
  --db_render_dirs ../output/${SC}_dino_32ds_a09_1 ../output/${SC}_dino_32ds_a09_2 ../output/${SC}_dino_32ds_a09_3 \
  --query_subdir language_features_dim32_dino_smooth_a09 --tag RAMEN_a09 \
  --out_json ../eval_result/mcnemar/RAMEN_DINO_a09.json 2>&1 | grep -E "chosen|multi-frame"
cd ..
$PY eval/mcnemar.py --clip_json eval_result/mcnemar/RAMEN_CLIP8d.json \
  --dino_json eval_result/mcnemar/RAMEN_DINO_a09.json --protocol any --tag RAMEN_a09_vs_CLIP
$PY eval/mcnemar.py --clip_json eval_result/mcnemar/RAMEN_DINO_a07.json \
  --dino_json eval_result/mcnemar/RAMEN_DINO_a09.json --protocol any --tag RAMEN_a09_vs_a07
$PY - << 'PYEOF'
import json
from scipy.stats import binomtest
def maj(p): return { (r['frameA'], r['obj_i']): r['mf_majority'] for r in json.load(open(p))['pairs'] }
d = maj('eval_result/mcnemar/RAMEN_DINO_a09.json')
for name, pj in [('CLIP8d', 'eval_result/adaptive/exp048_ramen_clip8d.json'), ('a07', 'eval_result/adaptive/exp048_ramen_dino_32ds_a07.json')]:
    c = maj(pj)
    b = sum(1 for k in c if c[k] == 1 and d[k] == 0)
    cc = sum(1 for k in c if c[k] == 0 and d[k] == 1)
    p = binomtest(min(b, cc), b + cc, 0.5).pvalue * 2 if b + cc else 1.0
    print(f'[RAMEN_a09 majority vs {name}] b={b} c={cc} p={min(p,1.0):.4f}')
PYEOF
rm -rf output/${SC}_dino_32ds_a09_1 output/${SC}_dino_32ds_a09_2 output/${SC}_dino_32ds_a09_3
echo "=== ramen a09 done $(date +%H:%M:%S)"
echo EXP054B_ALL_DONE
