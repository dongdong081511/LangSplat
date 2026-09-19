# LangSplat 多模态融合实验参数与结果记录

> 数据集: LERF teatime (177张图)
> 基线: LangSplat (CLIP ViT-B/16, laion2b_s34b_b88k)
> 评估指标: IoU chosen, Localization accuracy
> mask_thresh: 0.40 (teatime)

---

## 实验编号: EXP-001 — Baseline (纯CLIP)
- **分支**: develop
- **方案**: LangSplat 原版，SAM tile 级 CLIP 特征 (512维)
- **AE**: train.py 原版 autoencoder, 512→3
- **Gaussian训练**: 3 level (feature_level 1,2,3), 30000 iter, 从 chkpnt30000.pth restore
- **结果**:
  - IoU chosen: **0.6431**
  - Localization: **0.8983**

---

## 实验编号: EXP-002 — PCA Blend01 (90%CLIP + 10%DINOv2)
- **分支**: experiment/multires-C
- **方案**: numpy SVD 将 DINOv2 768→512, blend = 0.9*CLIP + 0.1*proj(DINOv2)
- **AE**: best_loss = 0.285
- **结果**:
  - IoU chosen: **0.5854** (-5.8% vs baseline)
  - Localization: **0.8814** (-1.7% vs baseline)
- **结论**: 简单混合稀释CLIP语义判别力

---

## 实验编号: EXP-003 — PCA Blend02 (80%CLIP + 20%DINOv2)
- **分支**: experiment/multires-C
- **方案**: numpy SVD 将 DINOv2 768→512, blend = 0.8*CLIP + 0.2*proj(DINOv2)
- **AE**: best_loss = 0.285
- **结果**:
  - IoU chosen: **0.5844** (-5.9% vs baseline)
  - Localization: **0.8644** (-3.4% vs baseline)
- **结论**: DINOv2比例越高，IoU和Localization下降越多

---

## 实验编号: EXP-004 — Cross-Attention 融合 (CLIP+DINOv2)
- **分支**: experiment/crossattn-mm
- **方案**: tile级 cross-attention 融合网络
  - 架构: self-attention + cross-attention + FFN
  - CLIP input dropout (0.3) 强制学习DINOv2特征
  - 输出: 512维 (CLIP兼容)
  - **关键修复**: 去掉全局 residual + 去掉 zero init (原设计导致网络只复制CLIP, loss=0)
- **融合网络训练**: 200 epochs, Loss 0.269→0.029, cos_sim 0.732→0.971
- **融合特征**: cos_sim=0.9806 (与CLIP的相似度)
- **AE**: best_loss = 0.1888 (优于blend02的0.285)
- **encode_dim3**: cos_sim=0.9349, 87.3% > 0.9
- **Gaussian训练**: 3 level, 30000 iter, 从 chkpnt30000.pth restore
  - Level 1: PSNR 31.73
  - Level 2: PSNR 31.73
  - Level 3: PSNR 31.73
- **结果**:
  - IoU chosen: **0.5351** (-10.8% vs baseline, -5.0% vs blend01)
  - Localization: **0.8814** (-1.7% vs baseline, 同blend01)
- **结论**:
  - cross-attention 融合网络虽然 AE 重建损失更低 (0.1888 vs 0.285)，但下游 IoU 反而更差
  - 融合网络 cos_sim=0.98 说明它主要在学习重建 CLIP 特征，但在此过程中丢失了语义判别力
  - DINOv2 特征即使通过 cross-attention 融合，仍然稀释 CLIP 语义判别力
  - 所有融合方案 (PCA blend, cross-attn) 均劣于 baseline，表明在 LERF teatime 开放词汇3D分割任务上，DINOv2 特征无增益

---

## 实验编号: EXP-005 — AE 瓶颈验证：baseline 6d
- **分支**: experiment/crossattn-mm
- **方案**: 与 baseline 相同（纯CLIP），但 AE 输出从 3d 改为 6d
  - 修改 rasterizer config.h NUM_CHANNELS_language_feature 6
  - 修改 gaussian_model.py / gaussian_renderer 6d
  - AE encoder: [256, 128, 64, 32, 6], decoder: [16, 32, 64, 128, 256, 256, 512]
- **AE**: best_loss（cos_sim=0.9399, 95.3% > 0.9）
- **结果**:
  - IoU chosen: **0.6285** (-1.5% vs baseline 3d)
  - Localization: **0.8644** (-3.4% vs baseline 3d)
- **结论**: 纯CLIP在6d下IoU和Loc反而下降，3d是CLIP的最优维度

---

## 实验编号: EXP-006 — AE 瓶颈验证：cross-attn fused 6d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-004 相同的cross-attn融合特征，但AE输出从3d改为6d
  - rasterizer + gaussian_model + renderer 全部改为6d
  - AE encoder: [256, 128, 64, 32, 6]
- **AE**: best_loss（cos_sim=0.9560, 98.7% > 0.9，远优于3d的87.3%）
- **Gaussian训练**: 3 level, 30000 iter
- **结果**:
  - IoU chosen: **0.6251** (vs 3d 0.5351 → **+9.0%提升**)
  - Localization: **0.9153** (vs 3d 0.8814 → **+3.4%提升**，超过baseline 3d 0.8983)
- **结论**:
  - **3d瓶颈是不成比例损害fused特征的根本原因**
  - 3d→6d: fused IoU +9.0%，而baseline IoU -1.5% → 6d对fused有选择性增益
  - fused 6d Localization 0.9153 > baseline 3d 0.8983 → DINOv2在足够维度下确实提升定位
  - fused 6d IoU 0.6251 ≈ baseline 3d 0.6431 (-1.8%)，差距从3d的-10.8%大幅缩小
  - **推翻之前结论**: DINOv2在足够AE维度下对融合有增益，之前的失败是3d瓶颈导致

---

## 实验编号: EXP-007 — AE 瓶颈验证：baseline 12d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-005 相同（纯CLIP），但 AE 输出从 6d 改为 12d
  - 修改 rasterizer config.h NUM_CHANNELS_language_feature 12
  - 修改 gaussian_model.py / gaussian_renderer 12d
  - AE encoder: [256, 128, 64, 32, 12], decoder: [16, 32, 64, 128, 256, 256, 512]
- **AE**: best_loss=0.14691803 (cos_sim=0.9500, 99.7% > 0.9)
- **Gaussian训练**: 3 level, 30000 iter, PSNR 31.73
- **结果**:
  - IoU chosen: **0.6660** (+2.3% vs baseline 3d, +3.8% vs baseline 6d)
  - Localization: **0.8814** (-1.7% vs baseline 3d, +1.7% vs baseline 6d)
- **结论**: baseline 在 12d 下 IoU 创新高（0.6660 > 3d 0.6431 > 6d 0.6285），说明 12d 是 CLIP 的更优维度；但 Loc 仍不及 3d

---

## 实验编号: EXP-008 — AE 瓶颈验证：cross-attn fused 12d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-004 相同的cross-attn融合特征，但AE输出从6d改为12d
  - rasterizer + gaussian_model + renderer 全部改为12d
  - AE encoder: [256, 128, 64, 32, 12]
- **AE**: best_loss=0.09224107 (cos_sim=0.9687, 100.0% > 0.9，远优于6d的98.7%)
- **Gaussian训练**: 3 level, 30000 iter, PSNR 31.73
- **结果**:
  - IoU chosen: **0.6609** (vs 6d 0.6251 → +3.6%提升，vs 3d 0.5351 → +12.6%提升)
  - Localization: **0.8644** (vs 6d 0.9153 → -5.1%下降)
- **结论**:
  - **12d下fused IoU继续提升**：3d 0.5351 → 6d 0.6251 → 12d 0.6609，单调上升
  - **fused 与 baseline 差距在 12d 几乎消失**：0.6609 vs 0.6660 (-0.5%)，3d时-10.8%，6d时-1.8%
  - 但 Localization 从 6d 的 0.9153 回落到 0.8644，12d 不是 Loc 的最优维度
  - DINOv2 融合在 IoU 上随维度增加持续逼近 baseline，印证 AE 瓶颈假设

---

## 实验编号: EXP-009 — InfoNCE 融合损失 12d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-004 相同的cross-attn融合网络，但损失函数改为 InfoNCE 对比学习
  - 损失: 跨模态 InfoNCE，fused 为 anchor，同 tile CLIP 为 positive，其他 tile CLIP 为 negative
  - temperature=0.07 (CLIP 默认值)
  - AE encoder: [256, 128, 64, 32, 12]
- **融合网络训练**: 200 epochs, cos_sim(fused, CLIP)=0.5810（大幅偏离CLIP，学到判别特征）
- **AE**: best_loss=0.24119715（远高于mse_cos的0.0922，因fused特征偏离CLIP重建困难）
- **encode_dim3**: cos_sim=0.8022, 仅5.0% > 0.9（重建质量差）
- **Gaussian训练**: 3 level, 30000 iter
- **结果**:
  - IoU chosen: **0.5270** (-13.9% vs baseline 12d, -20.9% vs mse_cos fused 12d)
  - Localization: **0.8475** (-3.9% vs baseline 12d, -1.9% vs mse_cos fused 12d)
- **结论**:
  - InfoNCE 对比学习使 fused 特征过度偏离 CLIP 空间 (cos_sim 0.58 vs 0.98)
  - 虽然"学到判别特征"，但偏离 CLIP 语义空间导致 AE 重建困难 (loss 0.241 vs 0.092)
  - 下游 IoU 大幅下降，甚至低于 3d fused (0.5351)
  - **对比学习不适合此任务**：CLIP 特征空间本身已是最优语义空间，过度偏离反而损害

---

## 实验编号: EXP-010 — MSE-only 融合损失 12d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-004 相同的cross-attn融合网络，但损失函数去掉 cos_sim 项，仅用 MSE
  - 损失: `mse_only_loss = F.mse_loss(pred, target)`
  - AE encoder: [256, 128, 64, 32, 12]
- **融合网络训练**: 200 epochs, cos_sim(fused, CLIP)=0.9796（与原版mse_cos 0.9806近似）
- **AE**: best_loss=0.08425621（低于mse_cos的0.0922，重建更好）
- **encode_dim3**: cos_sim=0.9711, 100.0% > 0.9（重建优秀）
- **Gaussian训练**: 3 level, 30000 iter
- **结果**:
  - IoU chosen: **0.6723** (+0.6% vs baseline 12d 0.6660，**首次超越baseline**！+1.1% vs mse_cos fused 12d 0.6609)
  - Localization: **0.9153** (+3.4% vs baseline 12d 0.8814，+5.1% vs mse_cos fused 12d 0.8644)
- **结论**:
  - **MSE-only 是最优损失函数**：去掉冗余的 cos_sim 项后，IoU 和 Loc 同时提升
  - cos_sim 项对 unit-normalized 输出与 MSE 近似等价 (MSE ≈ 2*(1-cos_sim))，属冗余约束
  - 去掉后网络更自由地学习融合特征，AE 重建更好 (0.084 vs 0.092)
  - **首个超越 baseline 的融合方案**：DINOv2 融合在 MSE-only + 12d 下确实有增益
  - 推翻"DINOv2 无增益"结论：关键在于损失函数设计和 AE 维度共同作用

---

## 实验编号: EXP-011 — baseline (纯CLIP) 16d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-007 相同（纯CLIP），但 AE 输出从 12d 改为 16d
  - 修改 rasterizer config.h NUM_CHANNELS_language_feature 12→16
  - 修改 gaussian_model.py / gaussian_renderer 硬编码 12→16
  - AE encoder: [256, 128, 64, 32, 16], decoder: [16, 32, 64, 128, 256, 256, 512]
- **AE**: best_loss=0.13770021（低于 12d 的 0.1469，重建更好）
- **encode_dim3**: cos_sim=0.9538, 99.7% > 0.9
- **Gaussian训练**: 3 level, 30000 iter, PSNR 31.73
- **结果**:
  - IoU chosen: **0.6401** (-2.6% vs baseline 12d 0.6660, -0.3% vs baseline 3d)
  - Localization: **0.8475** (-3.4% vs baseline 12d, -5.1% vs baseline 3d)
- **结论**: baseline 超过 12d 后开始退化（维度排序 12d > 3d > 16d > 6d），12d 是纯 CLIP 的最优维度

---

## 实验编号: EXP-012 — MSE-only 融合 16d
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-010 相同（cross-attn fused + mse_only），但 AE 输出从 12d 改为 16d
- **融合网络**: 复用 EXP-010 的 fusion_teatime_mseonly.pth（cos_sim=0.9796）
- **AE**: best_loss=0.08664520（与 12d 的 0.0843 相当）
- **encode_dim3**: cos_sim=0.9707, 100.0% > 0.9
- **Gaussian训练**: 3 level, 30000 iter, PSNR 31.73
- **结果**:
  - IoU chosen: **0.6698** (-0.25% vs fused 12d 0.6723，基本饱和)
  - Localization: **0.8814** (-3.4% vs fused 12d 0.9153)
  - vs baseline 16d: IoU **+3.0%**，Loc **+3.4%**
- **结论**:
  - **12d 确认为最优 AE 维度**：fused 在 16d 饱和略降，baseline 明显退化
  - 但 fused 的相对优势在 16d 扩大（+3.0% vs 12d 时的 +0.6%）：AE 维度升高时 baseline 退化快于 fused，fused 对 AE 维度更鲁棒
  - AE 重建 loss 更低（baseline 16d 0.1377 < 12d 0.1469）但下游 IoU 更差，再次验证"AE 重建好 ≠ 下游好"

---

## 汇总对比表

| 方案 | AE dim | IoU chosen | Localization | vs baseline 3d IoU | vs baseline 3d Loc |
|------|--------|-----------|---------------|---------------------|---------------------|
| baseline (CLIP) | 3 | 0.6431 | 0.8983 | - | - |
| baseline (CLIP) | 6 | 0.6285 | 0.8644 | -1.5% | -3.4% |
| baseline (CLIP) | 12 | 0.6660 | 0.8814 | +2.3% | -1.7% |
| baseline (CLIP) | 16 | 0.6401 | 0.8475 | -0.3% | -5.1% |
| blend01 (90%C+10%D) | 3 | 0.5854 | 0.8814 | -5.8% | -1.7% |
| blend02 (80%C+20%D) | 3 | 0.5844 | 0.8644 | -5.9% | -3.4% |
| cross-attn fused (mse_cos) | 3 | 0.5351 | 0.8814 | -10.8% | -1.7% |
| cross-attn fused (mse_cos) | 6 | 0.6251 | 0.9153 | -1.8% | +1.7% |
| cross-attn fused (mse_cos) | 12 | 0.6609 | 0.8644 | -0.5% | -1.7% |
| cross-attn fused (infonce) | 12 | 0.5270 | 0.8475 | -13.9% | -5.1% |
| **cross-attn fused (mse_only)** | **12** | **0.6723** | **0.9153** | **+2.9%** | **+1.7%** |
| cross-attn fused (mse_only) | 16 | 0.6698 | 0.8814 | +2.7% | -1.7% |

> 结论: **MSE-only 损失 + 12d 是最优组合**，IoU=0.6723 首次超越 baseline 12d (0.6660, +0.6%)，Loc=0.9153 远超 baseline (0.8814, +3.4%)。
> 16d 实验 (EXP-011/012) 确认 12d 为最优 AE 维度：fused 16d 饱和略降 (0.6698, -0.25pp)，baseline 16d 明显退化 (0.6401, -2.6pp)。
> 但 fused 对 AE 维度更鲁棒：16d 下 fused 优势扩大至 +3.0pp IoU / +3.4pp Loc（12d 时仅 +0.6pp IoU）。
> 维度规律: fused 单调上升至 12d 后饱和 (0.5351→0.6251→0.6609/0.6723→0.6698)；baseline 在 12d 达峰后退化 (0.6431→0.6285→0.6660→0.6401)。
> 损失函数对比: mse_only (0.6723) > mse_cos (0.6609) > infonce (0.5270)。
> InfoNCE 对比学习使特征过度偏离 CLIP 语义空间，AE 重建困难，下游 IoU 大幅下降。
> MSE-only 去掉冗余 cos_sim 约束后，网络更自由学习融合，AE 重建更好 (loss 0.084 vs 0.092)。
> 关键: DINOv2 融合的增益依赖 损失函数设计 + AE 维度 共同作用，三者缺一不可。
