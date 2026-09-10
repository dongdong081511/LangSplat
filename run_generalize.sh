#!/bin/bash
PROJECT=/home/xiedexia/project/LangSplat
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
MD=$PROJECT/hyper_parameter.md

run_baseline_eval() {
    local scene=$1 thresh=$2 eid=$3
    echo "[$eid] baseline eval: $scene (thresh=$thresh)..."
    cd $PROJECT/eval
    $PY evaluate_iou_loc.py --dataset_name $scene --feat_dir ../output --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result --mask_thresh $thresh --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --json_folder ../dataset/lerf_ovs/label > /tmp/eval_${eid}.txt 2>&1 || {
        echo "| $eid | $scene | baseline | FAILED | FAILED |" >> $MD; return
    }
    local iou=$(grep 'iou chosen:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    local loc=$(grep 'Localization accuracy:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    echo "[$eid] baseline: IoU=$iou, Loc=$loc"
    echo "| $eid | $scene | baseline | $iou | $loc | — |" >> $MD
}

run_scheme_a() {
    local scene=$1 thresh=$2 eid=$3
    local DATA=$PROJECT/dataset/lerf_ovs/$scene
    local GS_CKPT=$DATA/output/${scene}_-1/chkpnt30000.pth
    local NAME=${scene}_a

    echo ""
    echo "============================================"
    echo "  $eid: $scene 方案A (2级, thr=0.7)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    mkdir -p $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt
    cd $PROJECT/dataset/lerf_ovs/label
    [ -e $NAME ] || ln -s $scene $NAME
    cd $PROJECT

    echo "[$eid] preprocess..."
    $PY preprocess.py --dataset_path $DATA --multires_mode A --pyramid_levels 2 --cross_iou_thr 0.7 || {
        echo "| $eid | $scene | A1 | FAILED | FAILED |" >> $MD; return
    }

    cd $PROJECT/autoencoder
    echo "[$eid] AE train..."
    $PY train.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $scene | A1 | FAILED | FAILED |" >> $MD; return
    }
    cp $PROJECT/autoencoder/ckpt/$NAME/best_ckpt.pth $PROJECT/autoencoder/ckpt/$NAME/ae_ckpt/best_ckpt.pth

    echo "[$eid] dim3..."
    rm -rf $DATA/language_features_dim3
    $PY test.py --dataset_path $DATA --dataset_name $NAME --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 || {
        echo "| $eid | $scene | A1 | FAILED | FAILED |" >> $MD; return
    }

    cd $PROJECT
    echo "[$eid] 3DGS..."
    rm -rf output/${NAME}_1 output/${NAME}_2 output/${NAME}_3
    for lvl in 1 2 3; do
        $PY train.py -s $DATA -m output/$NAME --include_feature --feature_level $lvl --start_checkpoint $GS_CKPT || {
            echo "| $eid | $scene | A1 | FAILED | FAILED |" >> $MD; return
        }
        $PY render.py -m output/${NAME}_$lvl --include_feature --feature_level $lvl || {
            echo "| $eid | $scene | A1 | FAILED | FAILED |" >> $MD; return
        }
    done

    cd $PROJECT/eval
    echo "[$eid] eval..."
    $PY evaluate_iou_loc.py --dataset_name $NAME --feat_dir ../output --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result --mask_thresh $thresh --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --json_folder ../dataset/lerf_ovs/label > /tmp/eval_${eid}.txt 2>&1 || {
        echo "| $eid | $scene | A1 | FAILED | FAILED |" >> $MD; return
    }
    local iou=$(grep 'iou chosen:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    local loc=$(grep 'Localization accuracy:' /tmp/eval_${eid}.txt | tail -1 | grep -oP '[\d]+\.[\d]+')
    echo "[$eid] A1: IoU=$iou, Loc=$loc"
    echo "| $eid | $scene | A1 | $iou | $loc |" >> $MD
}

# 1. figurines baseline eval
run_baseline_eval figurines 0.45 G1-f
# 2. figurines 方案A
run_scheme_a figurines 0.45 G1-a
# 3. waldo_kitchen baseline eval
run_baseline_eval waldo_kitchen 0.40 G2-f
# 4. waldo_kitchen 方案A
run_scheme_a waldo_kitchen 0.40 G2-a

echo ""
echo "============================================"
echo "  ALL_GENERALIZE_DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
