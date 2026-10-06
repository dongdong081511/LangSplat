# IQ-Retrieve3D: Image-Query Cross-Frame 3D Retrieval Benchmark Protocol

> Version 0.1 (2026-10-04). Task-routing semantic fields for open-vocabulary 3D Gaussian Splatting.
> Text-query protocol (mIoU/mAcc, CLIP relevancy) follows LangSplat and is NOT covered here;
> this document specifies the **image-query** axis only.

## 1. Task Definition

Given a reconstructed 3D Gaussian scene with a per-Gaussian feature field and a query
**image patch** of an object observed in frame A, retrieve the same object in frame B:
the model must localize it by feature similarity over the rendered 2D feature map of frame B.

- **Query**: the SAM tile with the largest pixel overlap with the GT mask of the object in
  frame A, encoded by the same autoencoder (AE) used to train the field (query and database
  live in the same space — this mirrors the real pipeline).
- **Database**: rendered per-pixel feature maps of every other frame with GT, at 3 SAM levels.
- **Hit**: the similarity peak (after a 30x30 box-filter response smoothing) falls inside the
  same-label GT bbox of frame B.
- A query is scored against **all** frames B != A that contain a same-label object; the best
  (frame, level) candidate over all 3 levels is kept (chosen-level top1).

## 2. Scenes and Ground-Truth Statistics

Source: LERF-OVS (`teatime`, `figurines`, `waldo_kitchen`, `ramen`). GT = manually labeled
object masks/bboxes (COCO-style JSON per frame: `info{name,height,width}`, `objects[{category,
segmentation,bbox}]`).

| scene | GT frames | objects | labels | worst duplicate labels | evaluated pairs | mask_thresh |
|---|---|---|---|---|---|---|
| teatime | 6 | 62 | 14 | three cookies x7, plate x6, tea in a glass x6 | 61 | 0.40 |
| figurines | 4 | 57 | 21 | rubber duck with hat x4, pink ice cream x4, green apple x4 | 54 | 0.45 |
| waldo_kitchen | 5 | 29 | 18 | **knife x9**, plate x2, spoon x2, sink x2 | 13 | 0.40 |
| ramen | 7 | 80 | 14 | sake cup x8, chopsticks x7, egg x7, bowl x7, napkin x7 | 79 | 0.55 |

- `evaluated pairs` = (frameA, object) pairs that yield a valid query tile and at least one
  same-label frame B; the rest are excluded by the protocol.
- **Duplicate-label density is a first-class scene property** (e.g. waldo knife x9): under the
  first-bbox口径 a hit on any non-first duplicate is scored a miss. Report both口径 (Sec. 4).

## 3. Retrieval Protocol (frozen)

1. Load per-frame tile features `{frame}_f.npy` (AE-encoded, one row per tile id) and segment
   maps `{frame}_s.npy` (4 layers; layers 0/1/2 correspond to render levels 1/2/3).
2. Query tile: `argmax_t overlap(seg_level(obj.mask), t)`, feature `f_A[t]`, L2-normalized.
3. For each level (1..3) and each frame B with same-label objects:
   `sim = R_B @ q` per pixel, conv 30x30 box kernel, take argmax pixel.
4. Keep the global best (sim, level, frameB, pixel); hit if pixel inside GT bbox.
5. **first-bbox**: hit requires bbox[0] of the matched label list in frame B.
   **any-bbox**: hit if inside ANY same-label bbox (robust to duplicate labels).
6. Report: chosen-level top1 (first & any) + per-level top1 breakdown.

Reference implementation: `eval/eval_image_query.py` (`--out_json` dumps per-pair records;
join key `(frameA, obj_i)` for paired tests).

## 4. Metrics and Statistical Protocol

- Primary: chosen-level top1 under **both** first-bbox and any-bbox口径 (n/total).
- Paired significance: exact **McNemar test** per scene and pooled over scenes
  (`eval/mcnemar.py`, discordant counts b/c reported). Duplicate-label-heavy scenes MUST
  include the any-bbox test — first-bbox systematically underestimates fields that resolve
  duplicates correctly.
- Report per-level numbers when a field collapses on one level (e.g. coarse-level failure).

## 5. Baselines Released

All fields trained per LangSplat protocol (30k iters, 3 levels, restore from RGB ckpt).

| baseline | field | AE dim | note |
|---|---|---|---|
| CLIP (text-field) | LangSplat CLIP laion2b_s34b_b88k, per-scene optimal dim | teatime 8d / figurines 24d / waldo 8d / ramen 8d | text-query SOTA config reused as image-query baseline |
| DINO raw | DINOv2 ViT-S/14 tile features, 32d AE | 32d | no smoothing |
| DINO + global self-distill | alpha in {0.3, 0.5, 0.7} retrained | 32d | `eval/smooth_tiles.py` |
| DINO + adaptive (EXP-043) | per-tile gated alpha | 32d | `eval/smooth_tiles_adaptive.py` |

Reference numbers (chosen-level top1, first/any %) — full matrix in `paper_materials.md`:

| scene | CLIP | DINO a07 |
|---|---|---|
| teatime (n=61) | 81.97 / 86.89 | 91.80 / 93.44 |
| figurines (n=54) | 88.89 / 88.89 | 90.74 / 90.74 |
| ramen (n=79) | 74.68 / 82.28 | 78.48 / 84.81 |
| waldo (n=13) | 69.23 / 92.31 | 46.15 / 69.23 |

## 6. Reproducibility

- Env: conda `langsplat` (torch 2.x + CUDA), rasterizer compiled at
  `NUM_CHANNELS_language_feature = 32` for DINO fields (8d/24d for CLIP fields accordingly).
- Pipeline per scene: SAM preprocess -> tile feature extraction (CLIP / DINOv2) -> AE train
  (`autoencoder/train.py`, lr 1e-4) -> encode -> `train.py -m <out> --feature_level L
  --include_feature --start_checkpoint <rgb ckpt>` -> `render.py --include_feature` ->
  `eval/eval_image_query.py`.
- Released artifacts: per-pair JSONs (`eval_result/mcnemar/*.json`), GT label dirs,
  evaluation code. **Dataset images are NOT redistributed** (LERF license); run the standard
  LERF-OVS download + our preprocess.

## 7. Known Pitfalls (protocol notes)

1. Query features must be encoded by the SAME AE checkpoint generation as the field
   (encode mtime must be newer than ckpt mtime; AE reconstruction cos_sim alone does NOT
   guarantee downstream quality).
2. CLIP-field image queries must use the AE-encoded tile features (dim8 space), not raw
   512d CLIP embeddings (dimension mismatch crash).
3. Level render dir <-> seg layer mapping is fixed: render level k <-> seg layer k-1.
4. Never mix `_s.npy` seg tables across feature extractors (DINO preprocess re-runs SAM;
   its `_s.npy` is the only one aligned with `_f_dino.npy`).
5. first-bbox口径 under-counts duplicate-label objects; any-bbox can over-credit ambiguous
   regions — hence dual口径 is mandatory, not optional.

## 8. Roadmap

- [ ] Expand scenes: 3D-OVS, Replica-style indoor, more LERF scenes (target >= 8 scenes, n >= 30 pairs each).
- [ ] Third-party baselines: OpenGaussian (DINO 3D instance field), LERF relevance-map argmax,
      LangSplat official release weights where available.
- [ ] Leaderboard JSON schema + auto-report script (`eval/eval_image_query.py --out_json` ->
      `eval/mcnemar.py` -> markdown table).
- [ ] Human-verified GT audit for duplicate-label frames.

## 8. 重复标签分层协议（EXP-051）

检索歧义有两个正交维度，必须分开报告：

| 场景 | GT帧 | 实例 | 跨帧重复率 | 同帧聚集 max | mean extra/帧 | 等级 |
|---|---|---|---|---|---|---|
| teatime | 6 | 62 | 98% | 3 (hooves) | 0.50 | low-ambiguity |
| figurines | 4 | 57 | 95% | 2 | 0.25 | low-ambiguity |
| ramen | 7 | 80 | 99% | 2 | 1.29 | high-spread（跨帧广布） |
| waldo | 5 | 29 | 52% | **5 (knife)** | 1.40 | high-clustered（同帧聚集） |

三条实证结论（EXP-048 per-label 拆解，`eval_result/adaptive/exp048_*.json`）：

1. **跨帧重复不惩罚 3D 检索场**——ramen 跨帧重复 99%（sake cup ×8、chopsticks ×7）却是 DINO 场 majority 增益最大场景（+25.3pp）；重复标签跨帧分布是任务正常难度，凸显 per-frame 检索稳定性价值。
2. **同帧聚集被 majority 口径部分抵消**——waldo knife 每帧 5 实例同框，chosen/first 口径下重创 DINO 场（EXP-039 −23pp），但 majority 口径下 knife 双方 8/9 平手：多帧独立投票稀释了单帧歧义。
3. **小样本场景必须报 per-label 表**——waldo majority 残余差距全在 plate（n=2，0/2 vs 2/2）噪声级标签；teatime apple（−20%）、figurines green toy chair（0/2）同理。n<20 的场景级结论必须附 per-label 拆解。

发布包中的落地：`eval_image_query.py --out_json` 的 per-pair JSON 含 label 字段；分层分析脚本读 JSON 按 label 聚合即可复现第 3 节全部表格。
