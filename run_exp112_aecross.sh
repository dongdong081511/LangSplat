#!/bin/bash
# EXP-112: AE cross-scene generalization (3x3 donor-AE x eval-scene matrix, CLIP 8d, no-ens).
# Diagonal = same-scene AE (protocol sanity, must reproduce master-table baselines);
# off-diagonal = per-scene AE applied to another scene's rendered field (expect sharp drop,
# since the AE is trained per-scene on that scene's 2D CLIP features).
set -u
cd /home/xiedexia/project/LangSplat
PY=/home/xiedexia/.conda/envs/langsplat/bin/python

# donor AE -> ckpt path + decoder dims (latent dim 8 for all three)
declare -A AE_DIR=( [tea]=autoencoder/ckpt/teatime_8d/ae_ckpt/best_ckpt.pth
                    [ram]=ckpt/ramen_clip_8d/best_ckpt.pth
                    [wal]=ckpt/waldo_kitchen_clip_8d/best_ckpt.pth )
declare -A DEC=( [tea]="16 32 64 128 256 256 512"
                 [ram]="8 32 64 128 256 256 512"
                 [wal]="8 32 64 128 256 256 512" )
# eval scene -> render prefix + GT scene + mask_thresh (per-scene baseline thresholds)
declare -A B_RDR=( [tea]=teatime_8d [ram]=ramen_clip8d [wal]=waldo_kitchen_clip8d )
declare -A B_GT=(  [tea]=teatime   [ram]=ramen          [wal]=waldo_kitchen )
declare -A B_TH=(  [tea]=0.40      [ram]=0.55          [wal]=0.40 )

mkdir -p eval_result/aecross
for B in tea ram wal; do
  for A in tea ram wal; do
    ALIAS=${B}_ae${A}
    ln -sfn ${B_GT[$B]} dataset/lerf_ovs/label/$ALIAS
    for L in 1 2 3; do ln -sfn ${B_RDR[$B]}_$L output/${ALIAS}_$L; done
    echo "--- EXP112 $ALIAS (donor=$A eval=$B) $(date +%H:%M:%S)"
    (cd eval && $PY -u evaluate_iou_loc.py --feat_dir ../output --dataset_name $ALIAS \
        --clip_ae_ckpt ../${AE_DIR[$A]} \
        --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result/aecross \
        --mask_thresh ${B_TH[$B]} --encoder_dims 256 128 64 32 8 \
        --decoder_dims ${DEC[$A]} \
        --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k 2>&1 | grep -E "iou chosen|mIoU|Localization|mAcc" | tail -4)
  done
done
echo ALL_EXP112_DONE
