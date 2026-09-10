#!/bin/bash
# 方案A: 图像金字塔多分辨率 SAM 完整实验 (teatime)

PROJECT=/home/xiedexia/project/LangSplat
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
DATA=$PROJECT/dataset/lerf_ovs/teatime
GS_CKPT=$DATA/output/teatime_-1/chkpnt30000.pth
NAME=teatime_a

echo "============================================"
echo "  方案A: 图像金字塔 (teatime, 2级金字塔)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 0. 准备资源
mkdir -p $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt
cd $PROJECT/dataset/lerf_ovs/label
[ -e $NAME ] || ln -s teatime $NAME
cd $PROJECT

# 1. preprocess (方案A: 多分辨率 SAM)
echo "[A] Step 1/5: preprocess (multires_mode=A, pyramid_levels=2)..."
$PY preprocess.py --dataset_path $DATA --multires_mode A --pyramid_levels 2 || {
    echo "FAILED at preprocess"; exit 1
}

# 2. AE train
cd $PROJECT/autoencoder
echo "[A] Step 2/5: AE train..."
$PY train.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
    echo "FAILED at AE train"; exit 1
}
cp $PROJECT/autoencoder/ckpt/$NAME/best_ckpt.pth $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt/best_ckpt.pth

# 3. dim3
echo "[A] Step 3/5: generate dim3..."
rm -rf $DATA/language_features_dim3
$PY test.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
    echo "FAILED at dim3"; exit 1
}

# 4. 3DGS train + render (3 levels)
cd $PROJECT
echo "[A] Step 4/5: 3DGS train+render..."
rm -rf output/${NAME}_1 output/${NAME}_2 output/${NAME}_3
for lvl in 1 2 3; do
    $PY train.py -s $DATA -m output/$NAME --include_feature --feature_level $lvl --start_checkpoint $GS_CKPT || {
        echo "FAILED at 3DGS level $lvl"; exit 1
    }
    $PY render.py -m output/${NAME}_$lvl --include_feature --feature_level $lvl || {
        echo "FAILED at render level $lvl"; exit 1
    }
done

# 5. eval
cd $PROJECT/eval
echo "[A] Step 5/5: eval..."
$PY evaluate_iou_loc.py --dataset_name $NAME --feat_dir ../output --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result --mask_thresh 0.40 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --json_folder ../dataset/lerf_ovs/label > /tmp/eval_scheme_a.txt 2>&1 || {
    echo "FAILED at eval"; exit 1
}

# 提取结果
iou=$(grep 'iou chosen:' /tmp/eval_scheme_a.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
loc=$(grep 'Localization accuracy:' /tmp/eval_scheme_a.txt | tail -1 | grep -oP '[\d]+\.[\d]+')

echo ""
echo "============================================"
echo "  方案A 结果: IoU=$iou, Loc=$loc"
echo "  (baseline: IoU=0.6431, Loc=0.8983)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 追加到 hyper_parameter.md
echo "| A  | - | - | 方案A图像金字塔(2级) | $iou | $loc |" >> $PROJECT/hyper_parameter.md
