#!/bin/bash
PROJECT=/home/xiedexia/project/LangSplat
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
DATA=$PROJECT/dataset/lerf_ovs/teatime
GS_CKPT=$DATA/output/teatime_-1/chkpnt30000.pth
NAME=teatime_b
MD=$PROJECT/hyper_parameter.md

mkdir -p $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt
cd $PROJECT/dataset/lerf_ovs/label
[ -e $NAME ] || ln -s teatime $NAME
cd $PROJECT

run_exp() {
    local eid=$1 scales=$2 desc=$3
    echo ""
    echo "============================================"
    echo "  $eid: scales=$scales ($desc)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    cd $PROJECT
    echo "[$eid] preprocess..."
    $PY preprocess.py --dataset_path $DATA --multires_mode B --context_scales "$scales" || {
        echo "| $eid | $scales | FAILED | FAILED |" >> $MD; return
    }

    cd $PROJECT/autoencoder
    echo "[$eid] AE train..."
    $PY train.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $scales | FAILED | FAILED |" >> $MD; return
    }
    cp $PROJECT/autoencoder/ckpt/$NAME/best_ckpt.pth $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt/best_ckpt.pth

    echo "[$eid] dim3..."
    rm -rf $DATA/language_features_dim3
    $PY test.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $scales | FAILED | FAILED |" >> $MD; return
    }

    cd $PROJECT
    echo "[$eid] 3DGS..."
    rm -rf output/${NAME}_1 output/${NAME}_2 output/${NAME}_3
    for lvl in 1 2 3; do
        $PY train.py -s $DATA -m output/$NAME --include_feature --feature_level $lvl --start_checkpoint $GS_CKPT || {
            echo "| $eid | $scales | FAILED | FAILED |" >> $MD; return
        }
        $PY render.py -m output/${NAME}_$lvl --include_feature --feature_level $lvl || {
            echo "| $eid | $scales | FAILED | FAILED |" >> $MD; return
        }
    done

    cd $PROJECT/eval
    echo "[$eid] eval..."
    $PY evaluate_iou_loc.py --dataset_name $NAME --feat_dir ../output --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result --mask_thresh 0.40 --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --json_folder ../dataset/lerf_ovs/label > /tmp/eval_${eid}.txt 2>&1 || {
        echo "| $eid | $scales | FAILED | FAILED |" >> $MD; return
    }
    iou=$(grep 'iou chosen:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    loc=$(grep 'Localization accuracy:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    echo "[$eid] 结果: IoU=$iou, Loc=$loc"
    echo "| $eid | $scales | $iou | $loc |" >> $MD
}

run_exp B1 "1.0,1.5" "2 crops, 中等上下文"
run_exp B2 "1.0,2.0" "2 crops, 宽上下文"
run_exp B3 "1.0,1.5,2.0" "3 crops, 全范围"
run_exp B4 "1.0,1.25" "2 crops, 轻微上下文"
run_exp B5 "1.0,1.25,1.5" "3 crops, 渐进上下文"

echo ""
echo "============================================"
echo "  ALL_SCHEME_B_GRID_DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
