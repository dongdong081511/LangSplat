#!/bin/bash
# 方案A: 图像金字塔超参数网格 (A2-A6)

PROJECT=/home/xiedexia/project/LangSplat
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
DATA=$PROJECT/dataset/lerf_ovs/teatime
GS_CKPT=$DATA/output/teatime_-1/chkpnt30000.pth
MD=$PROJECT/hyper_parameter.md

mkdir -p $PROJECT/autoencoder/ckpt/teatime_a/ae_ckpt
cd $PROJECT/dataset/lerf_ovs/label
[ -e teatime_a ] || ln -s teatime teatime_a

run_exp() {
    local eid=$1 levels=$2 thr=$3 desc=$4
    NAME="teatime_a"
    echo ""
    echo "============================================"
    echo "  $eid: levels=$levels thr=$thr ($desc)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    # 1. preprocess
    cd $PROJECT
    echo "[$eid] preprocess..."
    $PY preprocess.py --dataset_path $DATA --multires_mode A --pyramid_levels $levels --cross_iou_thr $thr || {
        echo "| $eid | $levels | $thr | $desc | FAILED | FAILED |" >> $MD
        return
    }

    # 2. AE train
    cd $PROJECT/autoencoder
    echo "[$eid] AE train..."
    $PY train.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $levels | $thr | $desc | FAILED | FAILED |" >> $MD
        return
    }
    cp $PROJECT/autoencoder/ckpt/$NAME/best_ckpt.pth $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt/best_ckpt.pth

    # 3. dim3
    echo "[$eid] dim3..."
    rm -rf $DATA/language_features_dim3
    $PY test.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $levels | $thr | $desc | FAILED | FAILED |" >> $MD
        return
    }

    # 4. 3DGS train+render
    cd $PROJECT
    echo "[$eid] 3DGS..."
    rm -rf output/${NAME}_1 output/${NAME}_2 output/${NAME}_3
    for lvl in 1 2 3; do
        $PY train.py -s $DATA -m output/$NAME --include_feature --feature_level $lvl --start_checkpoint $GS_CKPT || {
            echo "| $eid | $levels | $thr | $desc | FAILED | FAILED |" >> $MD
            return
        }
        $PY render.py -m output/${NAME}_$lvl --include_feature --feature_level $lvl || {
            echo "| $eid | $levels | $thr | $desc | FAILED | FAILED |" >> $MD
            return
        }
    done

    # 5. eval
    cd $PROJECT/eval
    echo "[$eid] eval..."
    $PY evaluate_iou_loc.py --dataset_name $NAME --feat_dir ../output --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result --mask_thresh 0.40 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --json_folder ../dataset/lerf_ovs/label > /tmp/eval_${eid}.txt 2>&1 || {
        echo "| $eid | $levels | $thr | $desc | FAILED | FAILED |" >> $MD
        return
    }
    iou=$(grep 'iou chosen:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    loc=$(grep 'Localization accuracy:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    echo "[$eid] 结果: IoU=$iou, Loc=$loc"
    echo "| $eid | $levels | $thr | $desc | $iou | $loc |" >> $MD
}

run_exp A2 3 0.7 "3级金字塔, 默认NMS"
run_exp A3 2 0.5 "2级金字塔, 激进合并"
run_exp A4 2 0.8 "2级金字塔, 保守合并"
run_exp A5 3 0.5 "3级金字塔, 激进合并"
run_exp A6 3 0.8 "3级金字塔, 保守合并"

echo ""
echo "============================================"
echo "  ALL_SCHEME_A_GRID_DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
