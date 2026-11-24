# Task-Routing Semantic Fields for Open-Vocabulary 3D Gaussian Splatting

Minimal reproducible code for the paper's core claims: **query-modality routing
over two independent feature fields** (text query -> per-scene CLIP autoencoder
field; image query -> DINOv2 field), **cross-view self-distillation smoothing**,
a **multi-frame evidence-aggregation (majority) protocol** with paired
significance testing, and the supporting **attribution / robustness /
failure-taxonomy analyses**.

Built on top of [LangSplat](https://github.com/minghanqin/LangSplat) (Qin et
al., CVPR 2024); the shared 3DGS geometry stage, the tile-based preprocessing
and the official text-query protocol are inherited from it. Please cite both.

## What is here (mapping to the paper's contributions)

| Pipeline stage | Code |
|---|---|
| Tile feature preprocessing (SAM segments + CLIP + DINOv2/DINOv3/depth tiles) | `preprocess.py`, `mm_langsplat/extractors/` |
| Per-scene 2D-feature autoencoder (CLIP 8d/24d, DINOv2 32d) | `autoencoder/`, `mm_langsplat/encode_dim3.py` |
| Semantic 3DGS field training / rendering (multi-level) | `train.py`, `render.py`, `scene/`, `gaussian_renderer/`, `submodules/langsplat-rasterization` (vendored, with the language-feature channel shim) |
| Cross-view self-distillation smoothing (alpha blending of raw-field renders into 2D tile supervision) | `eval/smooth_tiles.py`, `run_exp085_e4a.sh` |
| **Router** (2D-entry advantage signal + variance-compression signal, tau1/tau2, Otsu re-calibration) | `eval/scene_router.py`, `eval/exp081_router_histmatch.py`, `eval/exp083_router_robust.py` |
| Official text-query evaluation (mIoU / mAcc, +/- prompt ensemble) | `eval/evaluate_iou_loc.py`, `eval/openclip_encoder.py` |
| Image-query evaluation (multi-level retrieval) | `eval/eval_image_query.py` |
| E1: exhaustive frame-pair attribution probe (2D teacher -> 3D field) | `eval/exp082_attribution.py`, `eval/exp082b_figurines_attr.py` |
| E4: alpha sweep, per-pair offset & peak-shift analysis | `eval/exp086_e4bc.py` |
| E5: teacher-swap (CLIP image encoder through the same field pipeline) | `eval/exp090_teacherswap.py` |
| Evaluation-criteria matrix + paired McNemar (first/any/majority/hit-rate) | `eval/mcnemar.py`, `eval/framelevel_mcnemar.py`, `eval/clip32d_mcnemar.py`, `eval/ludvig_restricted_mcnemar.py` |
| Cross-view offset distributions (per object x per frame pair) | `eval/exp110_consistency_dist.py` |
| AE cross-scene transfer 3x3 (per-scene AE necessity) | `run_exp112_aecross.sh` |
| DINOv2 backbone-scale transfer (B/14 -> L/14) | `run_exp113_l14.sh`, `eval/exp113_dino_backbone.py` |
| Failure taxonomy + scene-attribute correlations | `eval/exp114_failure_taxonomy.py` |
| E3: query perturbation robustness | `eval/exp084_query_robust.py` |
| Negative axes (depth teacher, DINOv3 collapse diagnosis) | `eval/depth_query_test.py`, `eval/dino3_failure_diag.py` |
| Full per-scene chain (train -> render -> eval, both fields + router) | `run_master_table.sh`, `run_dino_field.sh` |

`mm_langsplat/train_align.py`, `merge_dual.py`, `train_fusion.py`, `fusion/`
reproduce the dual-stream fusion and uniform-field ablations (negative results:
feature-level fusion degrades the CLIP stream).

## Environment

```bash
conda env create -f environment.yml   # PyTorch 2.0.1 + CUDA 11.8 lineage
pip install submodules/langsplat-rasterization submodules/simple-knn submodules/segment-anything-langsplat
```

Download to `ckpts/`:
- SAM ViT-H: `sam_vit_h_4b8939.pth`
- OpenCLIP weights are fetched automatically (`ViT-B-16`, `laion2b_s34b_b88k`).
- DINOv2 weights are fetched automatically via torch.hub (ViT-B/14 default;
  the backbone-scale experiment uses `dinov2_vitl14`).

## Data

We use the LERF-OVS and 3D-OVS datasets with per-frame labelme-style GT under
`dataset/<scene>/label/`. Run LangSplat's COLMAP preprocessing per scene
(`process.sh`), then:

```bash
# 1. tile features: CLIP + DINOv2 (+ SAM segment maps _s.npy)
python preprocess.py --dataset_path dataset/lerf_ovs/<scene> --use_dino

# 2. per-scene autoencoders (CLIP -> 8d/24d; DINOv2 -> 32d)
python autoencoder/train.py --dataset_path dataset/lerf_ovs/<scene> \
    --data_subdir language_features --dataset_name <scene>_clip_8d \
    --input_dim 512 --encoder_dims 256 128 64 32 8 \
    --decoder_dims 8 32 64 128 256 256 512
python autoencoder/train.py --dataset_path dataset/lerf_ovs/<scene> \
    --data_subdir language_features_dino --dataset_name <scene>_dino_32d \
    --input_dim 768 --encoder_dims 256 128 32 32 32 \
    --decoder_dims 32 128 256 256 768

# 3. compress 2D tiles to the latent dim
python -m mm_langsplat.encode_dim3 --dataset_path dataset/lerf_ovs/<scene> \
    --data_subdir language_features --out_subdir language_features_dim8 \
    --ae_ckpt ckpt/<scene>_clip_8d/best_ckpt.pth --input_dim 512 \
    --encoder_dims 256 128 64 32 8 --decoder_dims 8 32 64 128 256 256 512

# 4. full chain: 3-level field training, rendering, text + image eval, router
bash run_dino_field.sh            # DINOv2 32d field (example scene: teatime)
bash run_master_table.sh <scene> <mask_thresh> <port>   # CLIP 8d + DINO 32d + smooth + eval
```

`run_master_table.sh` patches the language-feature channel width (8/32) in the
vendored rasterizer and rebuilds it, then trains both fields from the shared
frozen-geometry checkpoint, renders, and evaluates. Note that the channel width
is a compile-time constant, so switching feature dims requires a rebuild (the
scripts do this automatically).

## Router

`eval/scene_router.py` computes the two routing signals per scene from the
already-extracted 2D tile features: (A) the 2D tile-retrieval advantage of the
DINOv2 teacher over CLIP, and (B) the cross-view variance compression of the
DINOv2 field. Decisions use tau1/tau2 calibrated on the four LERF scenes;
`eval/exp081_router_histmatch.py` re-calibrates thresholds unsupervised (Otsu)
and `eval/exp083_router_robust.py` runs leave-one-scene-out and frame-subsample
stability.

## Notes for reproduction

- The per-scene autoencoder is load-bearing: applying one scene's AE to another
  scene's rendered features collapses text-query mIoU by ~92% (3x3 matrix,
  `run_exp112_aecross.sh`), with the diagonal reproducing per-scene baselines
  bit-identically.
- All paired significance tests (McNemar) operate on per-query discordant
  pairs; the probe scripts (`exp082*`, `exp110`) write per-pair/per-object
  records so every number in the paper can be recomputed.
- Experiment-chain scripts write to `output/` and `eval_result/`; both are
  results directories and are not part of this release.

## License

Inherited from LangSplat (see `LICENSE.md`). The vendored
`submodules/langsplat-rasterization` carries its own license header.
