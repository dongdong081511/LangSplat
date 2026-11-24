# EXP-106 (P1-5): per-prompt text-query Pareto + localization-error records.
# Mirrors evaluate_iou_loc.py's single-stream loading path unmodified and adds a
# per-prompt record layer: chosen-level IoU (returned by activate_stream) plus a
# localization record replicating lerf_localization's loop (30x30 smoothing,
# per-level argmax, chosen-level-by-max) with a continuous bbox-distance error.
# The chosen level and the argmax are threshold-independent, so records are
# valid for any mask_thresh.

import os
import sys
import json
import glob
import time
import logging
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork

from evaluate_iou_loc import (
    activate_stream, eval_gt_lerfdata, get_logger, seed_everything)


def loc_prompt_records(valid_map, img_ann, positives):
    """Replicates lerf_localization per prompt; adds continuous bbox distance."""
    n_head, n_prompt, _, _ = valid_map.shape
    recs = []
    for k in range(n_prompt):
        select_output = valid_map[:, k]
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        avg_filtered = cv2.filter2D(select_output.cpu().numpy().transpose(1, 2, 0), -1, kernel)
        score_lvl = np.zeros(n_head)
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])
        sel = int(np.argmax(score_lvl))
        coords = coord_lvl[sel]
        boxes = img_ann[positives[k]]['bboxes'].reshape(-1, 4)
        hit = 0
        dists = []
        for box in boxes:
            x1, y1, x2, y2 = box
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            for c in coords:
                dx = max(xmin - c[0], 0, c[0] - xmax)
                dy = max(ymin - c[1], 0, c[1] - ymax)
                dists.append(float(np.hypot(dx, dy)))
                if xmin <= c[0] <= xmax and ymin <= c[1] <= ymax:
                    hit = 1
        recs.append({
            "prompt": positives[k], "chosen_lvl": sel, "loc_hit": hit,
            "loc_err_px": min(dists) if dists else None,
            "argmax_xy": [float(coords[0][0]), float(coords[0][1])] if len(coords) else None,
        })
    return recs


def main():
    parser = ArgumentParser(description="per-prompt text Pareto + loc-error records")
    parser.add_argument("--scene", type=str, required=True,
                        help="GT label scene name, e.g. teatime")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="render dataset name, e.g. teatime_8d / ramen_clip8d / figurines_3d")
    parser.add_argument("--feat_dir", type=str, default="../output")
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--json_folder", type=str, default="../dataset/lerf_ovs/label")
    parser.add_argument("--output_dir", type=str, default="../eval_result/pareto_locerr")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--mask_thresh", type=float, default=0.4,
                        help="only used by activate_stream's vis; records are theta-independent")
    parser.add_argument("--encoder_dims", nargs="+", type=int, required=True)
    parser.add_argument("--decoder_dims", nargs="+", type=int, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo", normalize=True, colormap_min=-1.0, colormap_max=1.0)

    feat_dir = [os.path.join(args.feat_dir, f"{args.dataset_name}_{i}",
                             "train/ours_None/renders_npy") for i in range(1, 4)]
    output_path = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(output_path, exist_ok=True)
    json_folder = os.path.join(args.json_folder, args.scene)
    logger = get_logger(f"pareto_{args.dataset_name}", log_level=logging.INFO)

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    logger.info(f"frames: {len(eval_index_list)}, shape: {image_shape}")

    total_feat_dim = args.encoder_dims[-1]
    compressed_sem_feats = np.zeros(
        (3, len(eval_index_list), *image_shape, total_feat_dim), dtype=np.float32)
    for i in range(3):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda f: int(os.path.basename(f).split(".npy")[0]))
        for j, idx in enumerate(eval_index_list):
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    clip_model = OpenCLIPNetwork(device, clip_model_type=args.clip_model,
                                 clip_model_pretrained=args.clip_pretrained,
                                 clip_n_dims=512, relev_temp=10.0)
    checkpoint = torch.load(args.ae_ckpt, map_location=device)
    model = Autoencoder(args.encoder_dims, args.decoder_dims, input_dim=512).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    records = []
    acc_num = 0
    total_prompts = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx + 1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = torch.from_numpy(compressed_sem_feats[:, j, ...]).float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            clip_model.dino_state = None
            flat = sem_feat.flatten(0, 2)
            use_half = flat.shape[0] > 2_000_000
            outs = []
            for c0 in range(0, flat.shape[0], 200_000):
                d = model.decode(flat[c0:c0 + 200_000])
                outs.append(d.half() if use_half else d)
            restored_feat = torch.cat(outs, dim=0).view(lvl, h, w, -1)

        img_ann = gt_ann[f'{idx}']
        prompts = list(img_ann.keys())
        clip_model.set_positives(prompts)

        c_iou_list, _ = activate_stream(restored_feat, rgb_img, clip_model, image_name,
                                        img_ann, thresh=args.mask_thresh,
                                        colormap_options=colormap_options)
        valid_map = clip_model.get_max_across(restored_feat)
        loc_recs = loc_prompt_records(valid_map, img_ann, prompts)
        assert len(loc_recs) == len(c_iou_list)
        for prompt, iou, rec in zip(prompts, c_iou_list, loc_recs):
            records.append({
                "frame_idx": idx, "prompt": prompt,
                "chosen_iou": float(iou),
                "chosen_lvl": rec["chosen_lvl"], "loc_hit": rec["loc_hit"],
                "loc_err_px": rec["loc_err_px"], "argmax_xy": rec["argmax_xy"],
            })
        acc_num += sum(r["loc_hit"] for r in loc_recs)
        total_prompts += len(prompts)

    # sanity: our replicated localization must match the official mAcc accounting
    official_acc = acc_num / max(total_prompts, 1)
    mean_iou = float(np.mean([r["chosen_iou"] for r in records]))
    mean_err = float(np.mean([r["loc_err_px"] for r in records
                              if r["loc_err_px"] is not None and r["loc_hit"] == 0]))
    with open(args.out_json, "w") as f:
        json.dump({"scene": args.scene, "dataset_name": args.dataset_name,
                   "n_frames": len(eval_index_list), "n_records": len(records),
                   "mIoU_check": round(mean_iou, 4),
                   "mAcc_check": round(official_acc, 4),
                   "mean_miss_err_px": round(mean_err, 2) if mean_err == mean_err else None,
                   "records": records}, f)
    logger.info(f"[{args.dataset_name}] records={len(records)} "
                f"mIoU_check={mean_iou:.4f} mAcc_check={official_acc:.4f} -> {args.out_json}")


if __name__ == "__main__":
    main()
