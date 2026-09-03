# 多分辨率金字塔增强 LangSplat 特征捕捉 方案设计

> 在 LangSplat 预处理 pipeline 中引入「多分辨率金字塔结构」以增强特征与信息捕捉。包含：表述纠错、技术调研、4 个放置位置子方案、网络架构、数学原理、代码实现与嵌入方式。

---

## 1. 背景与 Pipeline 现状

### 1.1 LangSplat 原始 pipeline（preprocess.py）

输入：多视角图片（`dataset/<scene>/images/` 下逐张读取）。核心流程：

```
for 每张图 img:
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(img)   # SAM 生成 4 个语义层次 mask
    seg_imgs, seg_map = mask2segmap(masks, img)                                  # 裁剪 mask 区域 → pad → resize 224×224
    clip_embed = model.encode_image(seg_imgs)                                   # CLIP 编码 → 512 维特征
    保存 language_features/XX_f.npy (特征) + XX_s.npy (分割图)
```

关键事实：
- SAM **逐张**处理图片（preprocess.py L123），非多视角联合输入。
- SAM 已有 4 个**语义层次**（default/s/m/l = 子部分/部分/整体），这是**语义粒度**，不是空间分辨率。
- CLIP 固定 224×224 输入（preprocess.py L40 `Resize((224,224))`）。
- mask 裁剪区域经 `pad_img`（补成正方形）→ `cv2.resize((224,224))`（preprocess.py L310）后送 CLIP。

### 1.2 后续阶段

`language_features/` → Autoencoder 降维到 3 维 → LangSplat 3DGS 训练（level 1/2/3）→ 渲染 → 评估。本增强作用于**第一阶段（预处理）**，产物仍为 `language_features/*.npy`，对下游透明兼容。

---

## 2. 表述纠错

| 原表述 | 问题 | 纠正 |
|---|---|---|
| "多视角的图片经过多分辨率金字塔...送入 SAM" | SAM 是逐张处理，非多视角联合输入 | 多分辨率金字塔作用于**单张图片**（生成同图不同分辨率版本），与"多视角融合"是两个正交概念 |
| "多分辨率金字塔...更丰富的特征" | 未区分"分辨率"与 LangSplat 已有的"语义层次" | LangSplat 的 default/s/m/l 是**语义粒度**；多分辨率金字塔是**空间分辨率**维度，二者正交可叠加 |
| "SAM 和 CLIP 之间放模块" | 未提 CLIP 固定 224 输入的约束 | 该位置多分辨率只能用 multi-crop + 特征聚合实现，不能直接送不同尺寸图 |
| （未提） | 漏了成本最低的放置点 | **CLIP 之后**（特征级跨尺度聚合）也是可行位置 |

---

## 3. 技术调研总结

### 3.1 经典多分辨率/多尺度架构

| 方法 | 核心思想 | 相关性 |
|---|---|---|
| **FPN**（2017） | 自底向上 + 自顶向下上采样 + 1×1 横向连接，逐层融合语义与细节 | 特征融合策略可直接借鉴 |
| **高斯/拉普拉斯金字塔** | 高斯 `G_l = Downsample(Blur(G_{l-1}))`；拉普拉斯 `L_l = G_l - Upsample(G_{l+1})` 保存高频细节 | 图像级多分辨率生成的数学基础 |
| **HRNet**（2020） | 多分辨率支路并行，全程保持多尺度，反复融合 | 并行多分支范式参考 |
| **PIIP**（2025, arXiv:2501.07783） | 高分辨率图用小网络、低分辨率用大网络，跨分支交互融合 | "参数倒置"：高分辨率分支用轻量网络平衡算力 |

### 3.2 CLIP 多分辨率方法（关键约束参考）

| 方法 | 做法 | 关键点 |
|---|---|---|
| **Dragonfly**（arXiv:2406.00977） | 低/中/高三分辨率，中高分辨率切 crops，分别过 CLIP，**mean-pooling 聚合 token** | mean-pooling 是有效的 token 归约；多分辨率捕获细粒度细节 |
| **MROVSeg**（arXiv:2408.14776） | 单 CLIP backbone，滑窗把高分辨率切成 224 patches，**Multi-Res Adapter** 融合多分辨率特征 | 单 backbone 避免额外骨干，Adapter 恢复空间几何 |
| **QLIP**（arXiv:2505.23004） | 四叉树内容自适应 patch，替代 CLIP 均匀网格，drop-in 替换无需重训 | 解决 CLIP 固定分辨率的"介观偏差"与"插值偏差" |

> **核心约束**：CLIP ViT 预训练于 224×224，直接改输入分辨率会性能崩塌。多分辨率必须通过 **multi-crop + 聚合** 或 **四叉树 patch** 实现，不能直接送不同尺寸图。

### 3.3 SAM 多分辨率方法

| 方法 | 做法 | 关键点 |
|---|---|---|
| **WSI-SAM**（arXiv:2403.09257） | 引入 HR（高分辨率）token 与 LR（低分辨率）token，dual mask decoder 在中间层融合；**SAM 冻结，仅加少量参数** | 多分辨率特征在 decoder 中间层融合，保留预训练知识 |
| SAM 原生 | 输入 resize 到 1024×1024，ViT-H 编码出 64×64×256 feature map；`crop_n_layers` 支持多尺度裁剪 | LangSplat 已用 `crop_n_layers=1`（preprocess.py L369） |

---

## 4. 方案设计：4 个放置位置

四个方案对应 pipeline 不同插入点，互不冲突，可做消融对比。推荐优先级：**A > B > C > D**。

```
图片 ──▶ [方案A: 图像金字塔多分辨率] ──▶ SAM ──▶ [方案B: mask多分辨率crop] ──▶ CLIP ──▶ [方案C: 特征跨尺度聚合] ──▶ language_features
                                          ↑(内部)
                                   [方案D: SAM encoder HR/LR token]
```

### 方案 A：SAM 之前 —— 图像金字塔多分辨率输入（推荐首选）

**位置**：`preprocess.py` 的 `sam_encoder` 函数之前，对单张图生成多分辨率版本，分别送 SAM，融合 mask。

**动机**：小物体在高分辨率下被 SAM 捕捉（点采样密度相对更高），大物体/上下文在低分辨率下被捕捉。LangSplat 现在只用单一分辨率，可能漏掉极端尺度物体。

**网络架构**：
```
输入 img (H×W)
   ├── 分辨率 r=1.0  → SAM → masks_1
   ├── 分辨率 r=0.5  → SAM → masks_0.5
   └── 分辨率 r=2.0  → SAM → masks_2.0  (对原图上采样或裁剪子区域)
        ↓
   mask 合并 + 跨尺度 NMS 去重（复用已有 mask_nms）
        ↓
   统一 seg_map + seg_imgs → CLIP
```

**输入输出**：
- 输入：单张 RGB 图 `img: (H,W,3)`
- 输出：与原 `sam_encoder` 一致的 `(seg_images, seg_maps)`，但 mask 来源跨多分辨率

**数学原理**：
- 高斯金字塔：`G_l(i,j) = Σ_{m,n} w(m,n) · G_{l-1}(2i+m, 2j+n)`，其中 `w` 为高斯核
- 跨尺度 mask 融合：对 `M = ∪_r M_r`，按 IoU 做层次化 NMS，保留各尺度代表性 mask
- 掩码评分：`score(m) = stability(m) · iou_pred(m)`，跨尺度取 max

**代价**：SAM encoder 是主要开销（ViT-H ~0.15s/图），3 分辨率约 3× 开销。可用 PIIP 思路优化：高分辨率分支降低 `points_per_side`。

### 方案 B：SAM-CLIP 之间 —— mask 区域多分辨率 CLIP 编码

**位置**：`preprocess.py` 的 `_embed_clip_sam_tiles`（L176），对每个 mask 裁剪区域做多分辨率 CLIP 编码后聚合。

**动机**：原 pipeline 把每个 mask 区域 resize 到 224 送一次 CLIP，丢失细节。用 Dragonfly 式多分辨率 crop + 聚合，保留细粒度属性。

**网络架构**：
```
seg_img (mask 裁剪区域)
   ├── resize 224×224 (全局) → CLIP → f_global
   ├── 切 2×2 crops (224) → CLIP → f_local_1..4
   └── 切 4×4 crops (224) → CLIP → f_detail_1..16  (可选, 代价高)
        ↓
   聚合: f_mask = MeanPool(f_global, f_local_*)  (Dragonfly 证明 mean-pool 有效)
   或加权: f_mask = α·f_global + β·MeanPool(f_local)
        ↓
   L2 归一化 → 512 维
```

**输入输出**：
- 输入：单个 mask 裁剪图 `seg_img: (h,w,3)`
- 输出：512 维 CLIP 特征（与原一致），但信息更丰富

**数学原理**：
- 多 crop 聚合：`f = (1/K) Σ_k CLIP(crop_k(seg_img))`，再 L2 归一化
- 加权变体：`f = Σ_k α_k · f_k / ‖Σ_k α_k · f_k‖`
- 尺度互补：全局 crop 捕语义，局部 crop 捕细节，均值池化等价于多视角证据累积

**代价**：每个 mask 的 CLIP 调用数从 1 增到 5（2×2）或 17（4×4）。可用 MROVSeg 的 shared-backbone + Adapter 思路降低。

### 方案 C：CLIP 之后 —— 特征级跨尺度聚合（成本最低）

**位置**：`preprocess.py` 的 `create` 函数（L113），对 CLIP 输出的 512 维特征做跨 level/跨尺度聚合。

**动机**：LangSplat 已有 4 个语义层次各自产生 CLIP 特征，当前是简单拼接（L141 `torch.cat`）。可用 FPN 式跨层次特征融合增强语义表达，且**不动 SAM/CLIP**，改动最小。

**网络架构**：
```
各层次 CLIP 特征: f_default, f_s, f_m, f_l  (各 N_i × 512)
   ↓
跨层次融合（FPN 式上采样对齐 + 横向连接）：
   f_fused = FPN([f_s, f_m, f_l])
   ↓
聚合到统一 seg_map
```

**输入输出**：
- 输入：4 个层次的 CLIP 特征列表
- 输出：融合后的 512 维特征（维度不变，下游兼容）

**数学原理**：
- FPN 融合：`F_l = Conv1×1(f_l)`；自顶向下 `P_l = Upsample(P_{l+1}) + F_l`；`P_l = Conv3×3(P_l)`
- 跨层次注意力：`Attn(Q=f_high, K,V=f_low)` 做语义-细节互补

**代价**：极低，仅做小特征图卷积/注意力，不改预训练模型。

### 方案 D：SAM encoder 内部 —— WSI-SAM 式 HR/LR token 融合

**位置**：`submodules/segment-anything-langsplat` 的 SAM image encoder 内部。

**动机**：WSI-SAM 证明在 SAM encoder 中间层引入 HR/LR token 融合，冻结 SAM 仅加少量参数即可提升多尺度分割。

**网络架构**：
```
img (HR) → SAM ViT 前若干层 → HR feature tokens
img (LR) → SAM ViT 前若干层 → LR feature tokens
   ↓
中间层融合: fused = FusionModule(HR_tokens, LR_tokens)
   ↓
SAM 后续层 + mask decoder
```

**约束**：需修改 `submodules/`（项目规则禁止改子模块，须用户明确同意），且 SAM encoder 是 CUDA 扩展，改动复杂。**仅作高级方案备选**。

**数学原理**：
- 双 token 融合：`fused = Concat(HR_t, LR_t)` + cross-attention；或门控 `g·HR_t + (1-g)·LR_t`
- WSI-SAM 用 dual decoder 在中间层联合学习同物体多分辨率特征

**代价**：中高。需训练融合模块（SAM 冻结），推理时 HR/LR 两次 encoder 前向。

---

## 5. 消融实验设计

| 实验 | 配置 | 对比目标 |
|---|---|---|
| Baseline | 原始 LangSplat（单一分辨率） | 基线指标 |
| A-only | 方案 A（3 分辨率送 SAM） | SAM 前多分辨率收益 |
| B-only | 方案 B（mask 多 crop CLIP） | CLIP 多分辨率收益 |
| C-only | 方案 C（特征级跨层次融合） | 不动预训练模型的收益 |
| A+B | A + B 叠加 | 两处多分辨率是否互补 |
| A+B+C | 全量 | 上限 |
| D | 方案 D | vs A（都作用于 SAM，对比外部多分辨率 vs 内部 token 融合） |

**评估指标**：LERF 4 场景（figurines/ramen/teatime/waldo_kitchen）的 mIoU 与 Localization Accuracy（论文对照：Overall 84.3 / 51.4）。

**建议**：先跑 A、C（改动小），再跑 B，最后按需上 D。

---

## 6. 代码实现与嵌入

### 6.1 方案 A 实现（图像金字塔多分辨率 SAM）

新增模块文件 `multires_pyramid.py`，在 `preprocess.py` 的 `sam_encoder` 调用前注入。遵循 Python 3.9 语法、中文注释。

```python
# multires_pyramid.py
"""多分辨率金字塔：为单张图生成多分辨率版本，供 SAM 多尺度捕捉。"""
import cv2
import numpy as np
from typing import List, Tuple


def build_gaussian_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
    """生成高斯金字塔。levels=3 返回 [原图, 0.5x, 0.25x]。"""
    pyramid = [image]
    for _ in range(levels - 1):
        blurred = cv2.GaussianBlur(pyramid[-1], (5, 5), sigmaX=1.0)
        pyramid.append(cv2.pyrDown(blurred))
    return pyramid


def build_multires_images(image: np.ndarray,
                          scales: Tuple[float, ...] = (1.0, 0.5, 2.0)) -> List[np.ndarray]:
    """按指定缩放因子生成多分辨率图像集合。

    scale<1 下采样(捕捉大物体/上下文), scale>1 上采样(捕捉小物体细节)。
    """
    h, w = image.shape[:2]
    results = []
    for s in scales:
        resized = cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        results.append(resized)
    return results


def fuse_masks_across_scales(list_of_mask_tuples,
                             iou_thr: float = 0.7,
                             score_thr: float = 0.1) -> Tuple:
    """跨尺度 mask 融合 + NMS 去重。

    list_of_mask_tuples: 每个分辨率下 (masks_default, masks_s, masks_m, masks_l)
    返回: 融合去重后的 (masks_default, masks_s, masks_m, masks_l)
    复用 preprocess.py 中已有的 mask_nms / masks_update。
    """
    from preprocess import masks_update  # 复用已有 NMS 逻辑
    levels = zip(*list_of_mask_tuples)  # 4 个 level 各自的跨尺度 mask 列表
    fused = ()
    for level_masks in levels:
        all_masks = []
        for scale_masks in level_masks:
            all_masks.extend(scale_masks)
        fused += (all_masks,)
    fused = masks_update(*fused, iou_thr=iou_thr, score_thr=score_thr)
    return fused
```

嵌入 `preprocess.py` 的 `sam_encoder`（L296）：

```python
from multires_pyramid import build_multires_images, fuse_masks_across_scales

def sam_encoder(image, use_multires: bool = True):
    image_rgb = cv2.cvtColor(image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

    if use_multires:
        # 方案 A: 多分辨率分别送 SAM, 再融合 mask
        multires_imgs = build_multires_images(image_rgb, scales=(1.0, 0.5, 2.0))
        all_scale_masks = []
        for res_img in multires_imgs:
            md, ms, mm, ml = mask_generator.generate(res_img)
            all_scale_masks.append((md, ms, mm, ml))
        masks_default, masks_s, masks_m, masks_l = fuse_masks_across_scales(all_scale_masks)
    else:
        masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image_rgb)
        masks_default, masks_s, masks_m, masks_l = masks_update(
            masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    # 以下逻辑不变: mask2segmap → seg_imgs → 返回
    # ...
```

### 6.2 方案 B 实现（mask 多分辨率 CLIP）

修改 `_embed_clip_sam_tiles`（L176）：

```python
def encode_mask_multires(clip_model, seg_img: torch.Tensor,
                         crops_per_side: int = 2) -> torch.Tensor:
    """对单个 mask 区域做多分辨率 CLIP 编码并聚合。

    seg_img: (B, 3, 224, 224) 已 pad+resize 的 mask 图
    返回: (B, 512) 聚合特征
    """
    # 全局特征
    f_global = clip_model.encode_image(seg_img)
    f_global = f_global / f_global.norm(dim=-1, keepdim=True)

    # 局部 crop 特征 (2x2)
    B, _, H, W = seg_img.shape
    step_h, step_w = H // crops_per_side, W // crops_per_side
    local_feats = []
    for i in range(crops_per_side):
        for j in range(crops_per_side):
            crop = seg_img[:, :, i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
            crop_224 = torch.nn.functional.interpolate(crop, size=(224, 224), mode='bilinear')
            f = clip_model.encode_image(crop_224)
            f = f / f.norm(dim=-1, keepdim=True)
            local_feats.append(f)
    # Mean-pooling 聚合 (Dragonfly 证明有效)
    f_local = torch.stack(local_feats).mean(dim=0)
    f_mask = (f_global + f_local) / 2.0
    f_mask = f_mask / f_mask.norm(dim=-1, keepdim=True)
    return f_mask
```

嵌入 `_embed_clip_sam_tiles`：

```python
def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode].to("cuda")
        with torch.no_grad():
            # 方案 B: 多分辨率编码
            clip_embed = encode_mask_multires(model, tiles, crops_per_side=2)
        clip_embed = clip_embed / clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    return clip_embeds, seg_map
```

### 6.3 方案 C 实现（特征级跨层次融合）

```python
import torch.nn as nn

class LevelFusionFPN(nn.Module):
    """跨语义层次特征融合 (FPN 式), 输入输出维度均为 512。"""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.lateral_default = nn.Conv1d(dim, dim, 1)
        self.lateral_s = nn.Conv1d(dim, dim, 1)
        self.lateral_m = nn.Conv1d(dim, dim, 1)
        self.lateral_l = nn.Conv1d(dim, dim, 1)
        self.fuse_conv = nn.Conv1d(dim, dim, 3, padding=1)

    def forward(self, feats: dict) -> torch.Tensor:
        """feats: {'default': (N,512), 's': (N,512), 'm': (N,512), 'l': (N,512)}"""
        fd = self.lateral_default(feats['default'].unsqueeze(-1))
        fs = self.lateral_s(feats['s'].unsqueeze(-1))
        fm = self.lateral_m(feats['m'].unsqueeze(-1))
        fl = self.lateral_l(feats['l'].unsqueeze(-1))
        fused = fd + fs + fm + fl
        fused = self.fuse_conv(fused).squeeze(-1)
        return fused / fused.norm(dim=-1, keepdim=True)
```

方案 C 改动最小，可作为快速验证 baseline。

### 6.4 命令行开关（统一控制）

在 `preprocess.py` 的 argparse（L348）增加开关：

```python
parser.add_argument('--multires_mode', type=str, default='none',
                    choices=['none', 'A', 'B', 'C', 'AB', 'ABC'],
                    help='多分辨率金字塔模式')
```

运行示例：
```bash
# 方案 A (图像金字塔多分辨率 SAM)
conda run -n langsplat python preprocess.py --dataset_path $DS --multires_mode A

# 方案 A+B
conda run -n langsplat python preprocess.py --dataset_path $DS --multires_mode AB
```

产物仍写入 `language_features/`，下游 Autoencoder/3DGS 训练命令不变。

---

## 7. 实施路径建议

1. **先跑方案 C**（改动最小，不动 SAM/CLIP），验证跨层次融合是否带来增益。
2. **再跑方案 A**（图像金字塔多分辨率 SAM），验证多尺度 mask 是否捕捉到原单一分辨率漏掉的物体。
3. **跑方案 B**（mask 多 crop CLIP），验证细粒度属性是否改善。
4. **消融组合** A+B、A+C、A+B+C，找最优组合。
5. 方案 D 仅在前三者收益饱和且确需深入 SAM 内部时考虑。

每个方案后用现有评估流程对照论文指标：
```bash
cd eval
python evaluate_iou_loc.py --dataset_name <scene> --feat_dir ../output \
    --ae_ckpt_dir ../autoencoder/ckpt --output_dir ../eval_result \
    --mask_thresh <场景最优阈值> --json_folder ../dataset/lerf_ovs/label
```

---

## 8. 风险与注意事项

| 风险 | 说明 | 缓解 |
|---|---|---|
| 预处理时间膨胀 | 方案 A 3× SAM 调用，方案 B 5× CLIP 调用 | PIIP 思路：高分辨率分支降 points_per_side；方案 B 用 shared-backbone |
| mask 数量爆炸 | 多分辨率产生更多 mask，维度膨胀 | 跨尺度 NMS 严格去重；限制 max masks |
| 与 LangSplat 训练兼容 | 增强后的 language_features 维度仍需 512 | 所有方案保证输出 512 维，下游透明 |
| 显存 | 多分辨率并发可能 OOM | 分辨率串行处理；方案 B 降低 crops_per_side |
| SAM 内部修改(方案D) | 改 submodules 违反项目规则，CUDA 扩展需重编译 | 仅作高级备选，须用户明确同意 |

---

## 9. 参考文献

- FPN: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
- HRNet: Wang et al., "Deep High-Resolution Representation Learning", CVPR 2020
- PIIP: Wang et al., "Parameter-Inverted Image Pyramid Networks", NeurIPS 2024 / arXiv:2501.07783
- Dragonfly: Thapa et al., arXiv:2406.00977
- MROVSeg: Zhu et al., arXiv:2408.14776
- QLIP: Chickering et al., arXiv:2505.23004
- WSI-SAM: Liu et al., arXiv:2403.09257
- LangSplat: Qin et al., CVPR 2024 / arXiv:2312.16084
