#!/bin/bash
# Master-table pipeline: one scene full chain (CLIP 8d field + DINO 32d field + alpha0.7 smooth)
# usage: bash run_master_table.sh <scene> <mask_thresh> <port_base>
set -e; set -o pipefail
SCENE=$1; THRESH=$2; PB=$3
SRC=/home/xiedexia/project/LangSplat/dataset/lerf_ovs/$SCENE
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
cd /home/xiedexia/project/LangSplat
echo "==== MASTER $SCENE start $(date +%H:%M:%S)"

# A. preprocess --use_dino (skip if done)
if ls $SRC/language_features_dino_raw/*_f_dino.npy >/dev/null 2>&1; then
  echo "A. skip preprocess (dino feats exist)"
else
  echo "--- A. preprocess --use_dino $(date +%H:%M:%S)"
  $PY -u preprocess.py --dataset_path $SRC --use_dino --save_subdir language_features_dino_raw 2>&1 | tail -3
fi
N_DINO=$(ls $SRC/language_features_dino_raw/*_f_dino.npy 2>/dev/null | wc -l)
echo "A. dino feats: $N_DINO"; [ "$N_DINO" -gt 100 ]

# B. reorganize language_features_dino
if [ ! -d $SRC/language_features_dino ]; then
$PY - <<'PYEOF'
import os, shutil, glob
SRC = os.environ['SRC']
raw = os.path.join(SRC, 'language_features_dino_raw'); clip = os.path.join(SRC, 'language_features'); dst = os.path.join(SRC, 'language_features_dino')
os.makedirs(dst, exist_ok=True); n = 0
for f in sorted(glob.glob(os.path.join(raw, '*_f_dino.npy'))):
    base = os.path.basename(f).replace('_f_dino.npy', '')
    shutil.copy(f, os.path.join(dst, base + '_f.npy'))
    # DINO preprocess recomputes SAM masks: its own _s is the only one aligned with _f_dino
    raw_s = os.path.join(raw, base + '_s.npy')
    assert os.path.exists(raw_s), f'missing {raw_s}'
    shutil.copy(raw_s, os.path.join(dst, base + '_s.npy'))
    n += 1
print('reorganized', n, 'frames'); assert n > 100
PYEOF
fi

# C. CLIP 8d AE + encode (official LangSplat dims)
if [ ! -f ckpt/${SCENE}_clip_8d/best_ckpt.pth ]; then
  echo "--- C. CLIP 8d AE $(date +%H:%M:%S)"
  $PY -u autoencoder/train.py --dataset_path $SRC --data_subdir language_features \
      --dataset_name ${SCENE}_clip_8d --input_dim 512 \
      --encoder_dims 256 128 64 32 8 --decoder_dims 8 32 64 128 256 256 512 2>&1 | tail -3
fi
ls ckpt/${SCENE}_clip_8d/best_ckpt.pth
if [ ! -d $SRC/language_features_dim8 ]; then
  $PY -u -m mm_langsplat.encode_dim3 --dataset_path $SRC --data_subdir language_features \
      --out_subdir language_features_dim8 --ae_ckpt ckpt/${SCENE}_clip_8d/best_ckpt.pth \
      --input_dim 512 --encoder_dims 256 128 64 32 8 --decoder_dims 8 32 64 128 256 256 512 2>&1 | tail -3
fi
ls $SRC/language_features_dim8 | wc -l

# D. DINO 32d AE + encode
if [ ! -f ckpt/${SCENE}_dino_32d/best_ckpt.pth ]; then
  echo "--- D. DINO 32d AE $(date +%H:%M:%S)"
  $PY -u autoencoder/train.py --dataset_path $SRC --data_subdir language_features_dino \
      --dataset_name ${SCENE}_dino_32d --input_dim 768 \
      --encoder_dims 256 128 32 32 32 --decoder_dims 32 128 256 256 768 2>&1 | tail -3
fi
ls ckpt/${SCENE}_dino_32d/best_ckpt.pth
if [ ! -d $SRC/language_features_dim32_dino ]; then
  $PY -u -m mm_langsplat.encode_dim3 --dataset_path $SRC --data_subdir language_features_dino \
      --out_subdir language_features_dim32_dino --ae_ckpt ckpt/${SCENE}_dino_32d/best_ckpt.pth \
      --input_dim 768 --encoder_dims 256 128 32 32 32 --decoder_dims 32 128 256 256 768 2>&1 | tail -3
fi
ls $SRC/language_features_dim32_dino | wc -l

# E. patch 8d + compile + CLIP GS x3
$PY - <<'PYEOF'
import re
DIM = 8
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', f'#define NUM_CHANNELS_language_feature {DIM}'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  f'torch.zeros((self._xyz.shape[0], {DIM}), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  f'torch.zeros((means3D.shape[0], {DIM}), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print('patched to 8')
PYEOF
if [ ! -f output/${SCENE}_clip8d_3/chkpnt30000.pth ]; then
  export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
  export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
  (cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_8d_${SCENE}.log 2>&1) || { echo COMPILE_FAIL; tail -20 /tmp/compile_8d_${SCENE}.log; exit 1; }
  echo "rasterizer compiled to 8d"
  for L in 1 2 3; do
    [ -f output/${SCENE}_clip8d_$L/chkpnt30000.pth ] && continue
    echo "--- 8d GS level $L $(date +%H:%M:%S)"
    $PY -u train.py -s $SRC -m output/${SCENE}_clip8d \
        --language_features_name language_features_dim8 --feature_level $L --include_feature \
        --start_checkpoint $SRC/output/${SCENE}_-1/chkpnt30000.pth \
        --port $((PB))$L 2>&1 | tail -2
    ls output/${SCENE}_clip8d_$L/chkpnt30000.pth
  done
fi

# F. patch 32d + compile + DINO GS x3
$PY - <<'PYEOF'
import re
DIM = 32
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', f'#define NUM_CHANNELS_language_feature {DIM}'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  f'torch.zeros((self._xyz.shape[0], {DIM}), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  f'torch.zeros((means3D.shape[0], {DIM}), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print('patched to 32')
PYEOF
if [ ! -f output/${SCENE}_dino_32d_3/chkpnt30000.pth ]; then
  export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
  export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
  (cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_32d_${SCENE}.log 2>&1) || { echo COMPILE_FAIL; tail -20 /tmp/compile_32d_${SCENE}.log; exit 1; }
  echo "rasterizer compiled to 32d"
  for L in 1 2 3; do
    [ -f output/${SCENE}_dino_32d_$L/chkpnt30000.pth ] && continue
    echo "--- 32d GS level $L $(date +%H:%M:%S)"
    $PY -u train.py -s $SRC -m output/${SCENE}_dino_32d \
        --language_features_name language_features_dim32_dino --feature_level $L --include_feature \
        --start_checkpoint $SRC/output/${SCENE}_-1/chkpnt30000.pth \
        --port $((PB+10))$L 2>&1 | tail -2
    ls output/${SCENE}_dino_32d_$L/chkpnt30000.pth
  done
fi

# F2. render original DINO field (G smooth_tiles needs these renders)
[ -d $SRC/language_features_dim32_dino_smooth_a07 ] || for L in 1 2 3; do
  [ -d output/${SCENE}_dino_32d_$L/train/ours_None/renders_npy ] && [ "$(ls output/${SCENE}_dino_32d_$L/train/ours_None/renders_npy | wc -l)" -gt 100 ] && continue
  echo "--- render dino_32d level $L $(date +%H:%M:%S)"
  $PY -u render.py -m output/${SCENE}_dino_32d_$L --include_feature 2>&1 | tail -1
done

# G. smooth alpha=0.7 (src=original dim32_dino feats, render_base=raw DINO field)
if [ ! -d $SRC/language_features_dim32_dino_smooth_a07 ]; then
  echo "--- G. smooth a07 $(date +%H:%M:%S)"
  (cd eval && $PY smooth_tiles.py --scene $SCENE --alpha 0.7 --iter_tag a07 2>&1 | tail -2)
fi
$PY - <<'PYEOF'
import numpy as np, os
SRC = os.environ['SRC']
o = np.load(f'{SRC}/language_features_dim32_dino/{sorted(os.listdir(SRC+"/language_features_dim32_dino"))[0]}')
s = np.load(f'{SRC}/language_features_dim32_dino_smooth_a07/{sorted(os.listdir(SRC+"/language_features_dim32_dino_smooth_a07"))[0]}')
d = float(np.abs(s - o).max()); print('smooth maxdiff', d); assert d > 1e-3, 'SMOOTH_FAILED_SILENT'
PYEOF
# reclaim disk: dino_32d renders are only consumed by smooth (~50GB/scene)
for L in 1 2 3; do rm -rf output/${SCENE}_dino_32d_$L/train/ours_None/renders_npy; done

# H. smooth-retrain GS x3 (still 32d, rasterizer already compiled)
for L in 1 2 3; do
  [ -f output/${SCENE}_dino_32ds_a07_$L/chkpnt30000.pth ] && continue
  echo "--- smooth GS level $L $(date +%H:%M:%S)"
  $PY -u train.py -s $SRC -m output/${SCENE}_dino_32ds_a07 \
      --language_features_name language_features_dim32_dino_smooth_a07 --feature_level $L --include_feature \
      --start_checkpoint $SRC/output/${SCENE}_-1/chkpnt30000.pth \
      --port $((PB+20))$L 2>&1 | tail -2
  ls output/${SCENE}_dino_32ds_a07_$L/chkpnt30000.pth
done

# I. render all fields — MUST render 32d models while 32d rasterizer is compiled,
#    then patch+compile back to 8d before rendering CLIP field (dim mismatch = CUDA illegal access)
for L in 1 2 3; do
  [ -d output/${SCENE}_dino_32ds_a07_$L/train/ours_None/renders_npy ] && [ "$(ls output/${SCENE}_dino_32ds_a07_$L/train/ours_None/renders_npy | wc -l)" -gt 100 ] && continue
  echo "--- render dino_32ds_a07 level $L $(date +%H:%M:%S)"
  $PY -u render.py -m output/${SCENE}_dino_32ds_a07_$L --include_feature 2>&1 | tail -1
done
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
$PY - <<'PYEOF'
import re
DIM = 8
edits = [
 ('submodules/langsplat-rasterization/cuda_rasterizer/config.h',
  r'#define NUM_CHANNELS_language_feature \d+', f'#define NUM_CHANNELS_language_feature {DIM}'),
 ('scene/gaussian_model.py',
  r'torch\.zeros\(\(self\._xyz\.shape\[0\], \d+\), device="cuda"\)',
  f'torch.zeros((self._xyz.shape[0], {DIM}), device="cuda")'),
 ('gaussian_renderer/__init__.py',
  r'torch\.zeros\(\(means3D\.shape\[0\], \d+\), dtype=opacity\.dtype, device=opacity\.device\)',
  f'torch.zeros((means3D.shape[0], {DIM}), dtype=opacity.dtype, device=opacity.device)'),
]
for f, pat, rep in edits:
    s = open(f).read(); s2, n = re.subn(pat, rep, s)
    assert n == 1, f'{f}: {n} matches'; open(f, 'w').write(s2)
print('patched to 8 for render')
PYEOF
(cd submodules/langsplat-rasterization && rm -rf build && $PY setup.py build_ext --inplace > /tmp/compile_8d_${SCENE}_r.log 2>&1) || { echo COMPILE_FAIL; tail -20 /tmp/compile_8d_${SCENE}_r.log; exit 1; }
echo "rasterizer recompiled to 8d for CLIP render"
for L in 1 2 3; do
  [ -d output/${SCENE}_clip8d_$L/train/ours_None/renders_npy ] && [ "$(ls output/${SCENE}_clip8d_$L/train/ours_None/renders_npy | wc -l)" -gt 100 ] && continue
  echo "--- render clip8d level $L $(date +%H:%M:%S)"
  $PY -u render.py -m output/${SCENE}_clip8d_$L --include_feature 2>&1 | tail -1
done
for F in clip8d dino_32ds_a07; do for L in 1 2 3; do
  echo "renders $F L$L: $(ls output/${SCENE}_${F}_$L/train/ours_None/renders_npy 2>/dev/null | wc -l)"
done; done

# J. text-query eval (mIoU/mAcc +- ensemble)
ln -sfn $SCENE dataset/lerf_ovs/label/${SCENE}_clip8d
cd eval
for MODE in "" "--prompt_ensemble"; do
  echo "--- text eval $SCENE mode=$MODE $(date +%H:%M:%S)"
  $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name ${SCENE}_clip8d \
      --clip_ae_ckpt ../ckpt/${SCENE}_clip_8d/best_ckpt.pth \
      --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
      --mask_thresh $THRESH --encoder_dims 256 128 64 32 8 \
      --decoder_dims 8 32 64 128 256 256 512 \
      --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k $MODE 2>&1 | grep -E "mIoU|mAcc"
done

# K. image-query eval (DINO smooth field vs CLIP 8d field)
echo "--- image eval $SCENE $(date +%H:%M:%S)"
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SCENE \
    --scene_root ../dataset/lerf_ovs/$SCENE \
    --db_render_dirs ../output/${SCENE}_dino_32ds_a07_1 ../output/${SCENE}_dino_32ds_a07_2 ../output/${SCENE}_dino_32ds_a07_3 \
    --query_subdir language_features_dim32_dino_smooth_a07 --tag DINO32d_a07 2>&1 | grep -v Warning
$PY -u eval_image_query.py --gt_dir ../dataset/lerf_ovs/label/$SCENE \
    --scene_root ../dataset/lerf_ovs/$SCENE \
    --db_render_dirs ../output/${SCENE}_clip8d_1 ../output/${SCENE}_clip8d_2 ../output/${SCENE}_clip8d_3 \
    --query_subdir language_features_dim8 --tag CLIP8d 2>&1 | grep -v Warning
cd ..
echo "==== MASTER $SCENE done $(date +%H:%M:%S) ===="
echo "ALL_EXP_DONE"
