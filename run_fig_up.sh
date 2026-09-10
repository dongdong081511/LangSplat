#!/bin/bash
# 方案A-Fig: figurines 上采样金字塔 (小物体适配)
PROJECT=/home/xiedexia/project/LangSplat
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
DATA=$PROJECT/dataset/lerf_ovs/figurines
GS_CKPT=$DATA/output/figurines_-1/chkpnt30000.pth
NAME=figurines_f
THRESH=0.45
MD=$PROJECT/hyper_parameter.md

mkdir -p $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt
cd $PROJECT/dataset/lerf_ovs/label
[ -e $NAME ] || ln -s figurines $NAME
cd $PROJECT

run_exp() {
    local eid=$1 ptype=$2 levels=$3 thr=$4 desc=$5
    echo ""
    echo "============================================"
    echo "  $eid: type=$ptype levels=$levels thr=$thr ($desc)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    cd $PROJECT
    echo "[$eid] preprocess..."
    $PY preprocess.py --dataset_path $DATA --multires_mode A --pyramid_levels $levels --cross_iou_thr $thr --pyramid_type $ptype || {
        echo "| $eid | $ptype | $levels | $thr | FAILED | FAILED |" >> $MD; return
    }

    cd $PROJECT/autoencoder
    echo "[$eid] AE train..."
    $PY train.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $ptype | $levels | $thr | FAILED | FAILED |" >> $MD; return
    }
    cp $PROJECT/autoencoder/ckpt/$NAME/best_ckpt.pth $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt/best_ckpt.pth

    echo "[$eid] dim3..."
    rm -rf $DATA/language_features_dim3
    $PY test.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $ptype | $levels | $thr | FAILED | FAILED |" >> $MD; return
    }

    cd $PROJECT
    echo "[$eid] 3DGS..."
    rm -rf output/${NAME}_1 output/${NAME}_2 output/${NAME}_3
    for lvl in 1 2 3; do
        $PY train.py -s $DATA -m output/$NAME --include_feature --feature_level $lvl --start_checkpoint $GS_CKPT || {
            echo "| $eid | $ptype | $levels | $thr | FAILED | FAILED |" >> $MD; return
        }
        $PY render.py -m output/${NAME}_$lvl --include_feature --feature_level $lvl || {
            echo "| $eid | $ptype | $levels | $thr | FAILED | FAILED |" >> $MD; return
        }
    done

    cd $PROJECT/eval
    echo "[$eid] eval..."
    $PY evaluate_iou_loc.py --dataset_name $NAME --feat_dir ../output --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result --mask_thresh $THRESH --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --json_folder ../dataset/lerf_ovs/label > /tmp/eval_${eid}.txt 2>&1 || {
        echo "| $eid | $ptype | $levels | $thr | FAILED | FAILED |" >> $MD; return
    }
    iou=$(grep 'iou chosen:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    loc=$(grep 'Localization accuracy:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    echo "[$eid] 结果: IoU=$iou, Loc=$loc"
    echo "| $eid | $ptype | $levels | $thr | $iou | $loc |" >> $MD
}

run_exp F1 up 2 0.7 "上采样2级 (1.0x+2.0x), 默认NMS"
run_exp F2 up 2 0.5 "上采样2级, 激进合并"
run_exp F3 up 2 0.8 "上采样2级, 保守合并"
run_exp F4 up 3 0.7 "上采样3级 (1.0x+2.0x+4.0x)"
run_exp F5 bidir 3 0.7 "双向 (0.5x+1.0x+2.0x)"

echo ""
echo "============================================"
echo "  ALL_FIG_UP_DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
