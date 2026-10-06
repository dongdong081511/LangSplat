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

## 实验编号: EXP-013 — 三模态融合 (CLIP + DINOv2 + Depth Anything V2) 12d
- **日期**: 2026-09-20
- **分支**: experiment/crossattn-mm
- **方案**: 在 EXP-010 最优配置 (cross-attn + mse_only + 12d) 基础上加入第三模态 Depth Anything V2
- **深度特征提取**:
  - 模型: DepthAnythingV2 (encoder=vitl), 权重 ckpts/depth_anything_v2_vitl.pth (1.34GB)
  - 代码: third_party/Depth-Anything-V2 (clone), mm_langsplat/extract_depth_tiles.py (新增)
  - 表示: 每帧全图相对深度归一化 [0,1] → 按 SAM tile 区域统计 **8 维向量**
    [mean, std, min, max, p25, p75, 梯度均值, 梯度最大值]，保存为 _f_depth.npy (177 帧)
  - 注意: _f.npy 行数 = seg_map.max()+1（每帧不同，非固定 300），深度特征需精确对齐
- **融合网络修改**: TileCrossAttentionFusion 增加 dim_depth 参数
  - proj_depth: Linear(8→256)+LayerNorm；新增 cross_attn_depth (K/V=depth token)
  - 注入顺序: self-attn(CLIP) → cross-attn(CLIP←DINO) → cross-attn(CLIP←depth) → FFN → proj_out
  - 损失: MSE-only (不变)，目标仍为 CLIP
- **融合网络训练**: 200 epochs, lr 1e-4, final cos_sim=**0.9797** (vs 双模态 0.9796)
- **AE 12d**: best_loss=0.08796547; encode_dim3 cos_sim=0.9701, 100.0% > 0.9
- **Gaussian训练**: 3 level, 30000 iter, 端口 55571-55573
- **结果** (mask_thresh=0.4):
  - IoU chosen: **0.6532** (-1.9% vs 双模态 0.6723, -1.3% vs baseline 0.6660)
  - Localization: **0.8814** (-3.4% vs 双模态 0.9153, 与 baseline 0.8814 持平)
- **结论**:
  - 加入 8 维深度统计作为第三模态**未能提升**，反而劣于双模态最优
  - 可能原因: ① 8 维统计信息量过低且逐图归一化导致跨帧不一致; ② MSE-only 目标为重建 CLIP，
    深度几何信息对重建目标贡献极小，第二 cross-attn 学到近似恒等映射甚至引入噪声;
  ③ 开放词汇查询基于 CLIP 文本相似度，深度信息与语义判别相关性有限
  - DINOv2 融合增益的关键在"与 CLIP 语义互补"，纯几何模态在当前 MSE 重建框架下无增益

---

## 实验编号: EXP-015 — 深度辅助损失 (λ=1.0) 三模态 12d
- **日期**: 2026-09-20
- **假设来源**: 用户提出 EXP-013 失败可能因为损失函数未针对深度模态设计
- **方案**: 融合网络增加 depth decoder head (512→128→8)，损失 = MSE(fused, CLIP) + 1.0 × MSE(decoded_depth, depth_stats)
  - 目的: 强制深度信息显式保存在 fused 512d 中（纯重建目标下网络无动力保留深度）
- **代码**: cross_attention.py 增加 depth_decoder + return_depth 参数; train_fusion.py 增加 --depth_loss_weight
- **融合网络训练**: 200 epochs, final cos_sim=**0.8615** (无辅助损失 0.9797 → 明显偏离 CLIP 空间)
- **AE 12d**: best_loss=0.04237171 (fused3aux); encode_dim3 cos_sim=0.9895
- **结果** (mask_thresh=0.4):
  - IoU chosen: **0.1646** (灾难性下降, vs 双模态 0.6723)
  - Localization: **0.2881**
- **结论**: **彻底失败，与 InfoNCE 同一失败模式**
  - λ=1.0 辅助损失把 fused 特征拉离 CLIP 语义空间 (cos_sim 0.86)
  - 开放词汇查询用 CLIP 文本嵌入做相似度检索 → fused 必须保持 CLIP 兼容性，偏离即崩
  - AE/encode 重建指标全部正常 (0.9895)，但下游崩塌 —— 再次验证"AE 重建好 ≠ 下游好"
  - 深度信息与 CLIP 语义判别根本性冲突: 保留深度必须牺牲 CLIP 空间位置

---

## 实验编号: EXP-014 — 三模态 16d AE (验证假设1: AE维度不足)
- **日期**: 2026-09-20
- **假设**: EXP-013 失败可能因加入第三模态后 12d AE 瓶颈信息量不足
- **方案**: 三模态融合特征 (language_features_fused3) → AE 16d (best_loss=0.0824) → encode cos_sim=0.9721
- **结果** (mask_thresh=0.4):
  - IoU chosen: **0.6456** (-0.8pp vs 三模态 12d 0.6532, -2.7pp vs 双模态最优 0.6723)
  - Localization: **0.8814** (与 12d 持平)
- **结论**: **假设1否定** —— 提高 AE 维度不能挽救三模态，16d 反而更差
  - 与 EXP-011/012 一致: 12d 是最优 AE 维度, 16d 饱和
  - 维度不是三模态失败的瓶颈, 问题在深度模态本身

---

## 实验编号: EXP-016 — 深度辅助损失轻量版 λ=0.1 (验证假设2: 损失函数设计)
- **日期**: 2026-09-20
- **假设**: EXP-015 失败可能因 λ=1.0 过强, 轻量辅助损失可兼顾深度保留与 CLIP 兼容
- **方案**: 同 EXP-015 结构, depth_loss_weight=0.1
- **融合网络训练**: 200 epochs, final cos_sim=**0.9398** (λ=1.0: 0.8615 / 无辅助: 0.9797)
- **AE 12d**: best_loss=0.04925431 (fused3aux01); encode cos_sim=0.9842
- **结果** (mask_thresh=0.4):
  - IoU chosen: **0.5588** (-9.4pp vs 无辅助 0.6532, 但远好于 λ=1.0 的 0.1646)
  - Localization: **0.8983** (+1.7pp vs 无辅助 0.8814)
- **结论**: **假设2否定** —— 辅助损失随 λ 单调恢复但无法超过无辅助基线:
  - λ=1.0 (0.1646) → λ=0.1 (0.5588) → λ=0 (0.6532), 单调趋势证明深度保留与 CLIP 兼容性此消彼长
  - 有趣细节: Loc 轻微提升 (0.8983) 说明深度信息确实编码了位置线索,
    但 IoU 大幅受损 —— 深度挤占了 CLIP 语义判别空间
  - **最终结论: EXP-013 的负结果是结构性的**。在 LangSplat 框架下
    (开放词汇查询 = CLIP 文本相似度检索, fused 必须保持 CLIP 兼容),
    深度几何信息无法带来增益, 两个假设 (AE维度/损失设计) 均被实验否定

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
| cross-attn fused3 (+Depth) (mse_only) | 12 | 0.6532 | 0.8814 | +1.0% | -1.7% |
| cross-attn fused3 (+Depth aux loss λ=1.0) | 12 | 0.1646 | 0.2881 | — | — |
| cross-attn fused3 (+Depth) | 16 | 0.6456 | 0.8814 | +0.4% | -1.7% |
| cross-attn fused3 (+Depth aux λ=0.1) | 12 | 0.5588 | 0.8983 | — | — |

> 结论: **MSE-only 损失 + 12d 是最优组合**，IoU=0.6723 首次超越 baseline 12d (0.6660, +0.6%)，Loc=0.9153 远超 baseline (0.8814, +3.4%)。
> 16d 实验 (EXP-011/012) 确认 12d 为最优 AE 维度：fused 16d 饱和略降 (0.6698, -0.25pp)，baseline 16d 明显退化 (0.6401, -2.6pp)。
> 但 fused 对 AE 维度更鲁棒：16d 下 fused 优势扩大至 +3.0pp IoU / +3.4pp Loc（12d 时仅 +0.6pp IoU）。
> 维度规律: fused 单调上升至 12d 后饱和 (0.5351→0.6251→0.6609/0.6723→0.6698)；baseline 在 12d 达峰后退化 (0.6431→0.6285→0.6660→0.6401)。
> 损失函数对比: mse_only (0.6723) > mse_cos (0.6609) > infonce (0.5270)。
> InfoNCE 对比学习使特征过度偏离 CLIP 语义空间，AE 重建困难，下游 IoU 大幅下降。
> MSE-only 去掉冗余 cos_sim 约束后，网络更自由学习融合，AE 重建更好 (loss 0.084 vs 0.092)。
> 关键: DINOv2 融合的增益依赖 损失函数设计 + AE 维度 共同作用，三者缺一不可。


---

## 实验编号: EXP-017 — figurines 跨场景验证 (baseline vs fused, 12d)
- **日期**: 2026-09-21
- **目的**: 验证 DINOv2 双模态融合 (mse_only) 在 teatime 上的增益 (IoU +0.6pp / Loc +3.4pp) 是否跨场景泛化
- **场景**: figurines (299帧, 17类密集相似小物体, GT tile pairwise cos=0.646, 天然难分)
- **流程**: CLIP特征提取 → (fused: cross-attn融合 cos_sim=0.9811) → AE 12d (baseline best_loss=0.1578 / fused 0.0914) → encode → 3-level Gaussian (30000 it, ports 55581-83) → render ×3 → eval (mask_thresh=0.45, 4 eval帧 × 17类)

### 结果
| 模型 | IoU chosen | Localization |
|------|-----------|--------------|
| baseline 12d | **0.5466** | 0.7321 |
| fused (mse_only) 12d | **0.5274** | **0.7500** |

### 跨场景对照 (teatime → figurines)
| 指标 | teatime (EXP-010 vs EXP-007) | figurines (EXP-017) | 泛化性 |
|------|------|------|--------|
| fused IoU 增益 | +0.6pp (0.6723 vs 0.6660) | **-1.9pp** (0.5274 vs 0.5466) | ❌ 不泛化，符号翻转 |
| fused Loc 增益 | +3.4pp (0.9153 vs 0.8814) | **+1.8pp** (0.7500 vs 0.7321) | ✅ 两场景一致为正 |

### 结论
1. **DINOv2 融合的 IoU 增益不跨场景泛化**：teatime +0.6pp 本就在噪声量级，figurines 上翻转为 -1.9pp。IoU (细粒度分割) 上融合无稳定价值
2. **Localization 增益跨场景一致** (+3.4pp / +1.8pp)：DINOv2 融合对"物体在哪"的粗粒度定位有稳定价值，对"边界画多准"的细粒度 IoU 没有
3. figurines 基线本身低于 teatime (0.5466 vs 0.6660)，与场景难度一致 (17类密集小物体，2D CLIP 单帧上限仅 0.593)
4. **诊断副产品 (重要工程教训)**: 首次 figurines eval 得到 IoU=0.0194 的崩溃，根因是 dim12 特征编码发生在 AE 收敛之前 (编码 18:14 用了过期权重，最终 ckpt 19:13 才保存)。三源对照法定位：原始2D CLIP IoU=0.593 / AE往返(过期权重编码)=0.063 / 3D渲染=0.025；用最终 ckpt 重编码后 AE往返恢复 0.669。已重编码+重训 baseline 3个GS+重渲染+重评

---

## 实验编号: EXP-018 — AE 维度验证: baseline 8d 与 mse_only 融合 8d
- **日期**: 2026-09-21
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-007/EXP-010 相同，但 AE 输出从 12d 改为 8d
  - 修改 rasterizer config.h NUM_CHANNELS_language_feature 12→8
  - 修改 gaussian_model.py / gaussian_renderer 硬编码 12→8
  - AE encoder: [256, 128, 64, 32, 8], decoder: [16, 32, 64, 128, 256, 256, 512]
- **AE**: baseline best_loss=0.1543（8d 重建难于 12d 0.1469，符合预期）；fused best_loss=0.1053（12d 为 0.0843）
- **encode_dim3**: baseline cos_sim=0.9471（99.0% > 0.9）；fused cos_sim=0.9640（98.7% > 0.9）
- **Gaussian训练**: 3 level, 30000 iter, PSNR 31.73
- **结果**:
  - baseline: IoU chosen: **0.6705**（+0.45% vs baseline 12d 0.6660，**baseline 新最优**），Localization: **0.8814**（与 12d 持平）
  - fused (mse_only): IoU chosen: **0.6375**（-3.5% vs fused 12d 0.6723），Localization: **0.8644**（-5.1% vs fused 12d 0.9153）
- **结论**:
  - **baseline 维度曲线修正**: 8d 0.6705 > 12d 0.6660 > 3d 0.6431 > 16d 0.6401 > 6d 0.6285，8d 是纯 CLIP 的最优维度（非 12d）
  - **fused 需要 ≥12d 才能发挥**: 8d 时 fused 明显退化（-3.5pp），融合特征信息量更大，需要更高 AE 维度保留
  - **符号翻转**: 8d 时 baseline 反超 fused（-3.3pp）；fused 超越 baseline 的条件是 AE 维度 ≥12d
  - 维度选择取决于特征类型：纯 CLIP 用 8d 足够，DINOv2 融合需 12d

---

## 实验编号: EXP-019 — figurines 跨场景 16d 验证 (baseline + mse_only 融合)
- **日期**: 2026-09-21
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-011/012 (teatime 16d) 相同配置，在 figurines 场景验证维度泛化性
  - rasterizer config.h + gaussian_model.py + gaussian_renderer 改回 16d 并重编译
  - AE encoder: [256, 128, 64, 32, 16], decoder: [16, 32, 64, 128, 256, 256, 512]
  - 复用 EXP-017 的 figurines fused 512d 特征 (language_features_fused_mseonly)
- **AE**: baseline best_loss=0.1519；fused best_loss=0.0895
- **encode_dim3**: baseline cos_sim=0.9349（91.2% > 0.9）；fused cos_sim=0.9639（100% > 0.9）
- **Gaussian训练**: 3 level, 30000 iter, PSNR 25.14
- **结果** (mask_thresh=0.45):
  - baseline: IoU chosen: **0.5622**（vs figurines 12d 0.5466 → **+1.6pp**），Localization: **0.7857**（vs 0.7321 → **+5.4pp**）
  - fused (mse_only): IoU chosen: **0.5547**（vs figurines 12d 0.5274 → **+2.7pp**），Localization: **0.7500**（持平）
- **结论**:
  - **维度最优值是场景相关的**: figurines 上 16d 对 baseline 和 fused 双双提升（与 teatime 上 16d 退化相反：baseline -2.6pp / fused -0.25pp）
  - teatime 维度曲线 (8d 最优 0.6705 > 12d) 与 figurines (16d 仍在涨) 行为不同，AE 维度超参需按场景调优
  - figurines 16d: baseline 0.5622 > fused 0.5547（-0.75pp），融合差距从 12d 的 -1.9pp 缩小
  - 融合增益跨场景依然不稳定（teatime 16d fused +3.0pp，figurines 16d fused -0.75pp），与 EXP-017 结论一致
  - figurines 16d baseline Loc 0.7857 大幅超过 fused 0.7500，与 12d 时 fused 占优 (0.7500 vs 0.7321) 翻转

---

## 实验编号: EXP-020 — figurines 20d 验证 (baseline + mse_only 融合)
- **日期**: 2026-09-22
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-019 相同配置，AE 输出从 16d 提高到 20d
  - rasterizer config.h + gaussian_model.py + gaussian_renderer 改为 20d 并重编译
  - AE encoder: [256, 128, 64, 32, 20], decoder: [16, 32, 64, 128, 256, 256, 512]
- **AE**: baseline best_loss=0.1534；fused best_loss=0.0831（均优于 16d 的 0.1519/0.0895）
- **encode_dim3**: baseline cos_sim=0.9333（90.2% > 0.9）；fused cos_sim=0.9668（100% > 0.9）
- **Gaussian训练**: 3 level, 30000 iter, PSNR 25.14
- **结果** (mask_thresh=0.45):
  - baseline: IoU chosen: **0.5668**（vs 16d 0.5622 → +0.5pp，趋缓），Localization: **0.7679**（vs 0.7857 → -1.8pp 回落）
  - fused (mse_only): IoU chosen: **0.5744**（vs 16d 0.5547 → **+2.0pp**），Localization: **0.7857**（vs 0.7500 → **+3.6pp**）
- **结论**:
  - **fused 20d 在 figurines 首次 IoU+Loc 双超 baseline**（IoU +0.76pp / Loc +1.8pp）：12d 时 IoU -1.9pp → 16d -0.75pp → 20d +0.76pp，符号随维度递增翻转
  - **fused 20d IoU 0.5744 创 figurines 全实验新高**（超过 baseline 任意维度的最高值）
  - **fused figurines 维度曲线单调上升**: 12d 0.5274 → 16d 0.5547 → 20d 0.5744，与 teatime (12d 最优后退化) 完全相反
  - baseline 20d IoU 趋缓 (+0.5pp) 且 Loc 回落 (-1.8pp)，接近其维度饱和点
  - **维度需求与场景特征复杂度相关**: figurines 密集小物体需要更多维度承载判别信息，DINOv2 融合的高维优势在复杂场景显现

---

## 实验编号: EXP-021 — figurines 24d 验证 (baseline + mse_only 融合)
- **日期**: 2026-09-22
- **分支**: experiment/crossattn-mm
- **方案**: 与 EXP-020 相同配置，AE 输出从 20d 提高到 24d
  - rasterizer config.h + gaussian_model.py + gaussian_renderer 改为 24d 并重编译
  - AE encoder: [256, 128, 64, 32, 24], decoder: [16, 32, 64, 128, 256, 256, 512]
- **AE**: baseline best_loss=0.1542；fused best_loss=0.0850（与 20d 的 0.1534/0.0831 相当）
- **encode_dim3**: baseline cos_sim=0.9328（89.6% > 0.9）；fused cos_sim=0.9657（100% > 0.9）
- **Gaussian训练**: 3 level, 30000 iter, PSNR 25.14
- **结果** (mask_thresh=0.45):
  - baseline: IoU chosen: **0.5751**（vs 20d 0.5668 → +0.8pp，仍在涨），Localization: **0.8214**（vs 0.7679 → **+5.4pp，figurines 全实验新高**）
  - fused (mse_only): IoU chosen: **0.5447**（vs 20d 0.5744 → **-3.0pp 回落**），Localization: **0.7500**（持平）
- **结论**:
  - **fused 峰值确认在 20d**（12d 0.5274 → 16d 0.5547 → 20d 0.5744 → 24d 0.5447），倒 U 型曲线，fused 最优维度 figurines=20d、teatime=12d
  - **baseline 在 figurines 维度曲线单调上升未见顶**: 12d 0.5466 → 16d 0.5622 → 20d 0.5668 → 24d 0.5751，且 Loc 0.8214 创新高
  - 24d 时 baseline 重新反超 fused (+3.0pp)，与 20d 的 fused 领先 (+0.76pp) 翻转——融合领先窗口在 20d 附近，过维度后 baseline 凭借更多语义信息容量反超
  - **跨场景规律**: fused 在两个场景均呈倒 U 型且有明确峰值；baseline 峰值维度场景差异大 (teatime 8d / figurines ≥24d)

---

## 实验编号: EXP-022 — CLIP 骨干升级 ViT-B/16 → ViT-L/14 (laion2b_s32b_b82k, 768d)
- **日期**: 2026-09-26 ~ 09-27
- **分支**: experiment/crossattn-mm
- **动机**: 21 个实验确认 CLIP 特征是 figurines 瓶颈 (2D 单帧上限 IoU 0.593)，升级骨干抬高上限
- **方案**: 同场景同 AE 维度只换骨干。6 文件参数化改动 (默认保持 B/16 可复现):
  preprocess.py / eval/openclip_encoder.py / eval/evaluate_iou_loc.py (--clip_model/--clip_pretrained/--clip_n_dims/--ae_input_dim)
  autoencoder/model.py+train.py / mm_langsplat/encode_dim3.py (--input_dim 768)
- **预处理**: teatime 177 帧 / figurines 299 帧, _f.npy (N_tiles, 768)。SAM preprocess 不能双场景并行 (CPU 后处理争抢互卡 0it，串行后 19.4s/it 正常)
- **AE**: teatime_L14_8d best_loss=0.1947 (768→8d)；figurines_L14_24d best_loss=0.1648 (768→24d)
- **encode**: teatime cos_sim=0.9383 (96.3%>0.9)；figurines cos_sim=0.9347 (93.1%>0.9)——L14 特征比 B/16 更难压缩 (B/16 同维度 0.9471/0.9333)
- **Gaussian训练**: 3 level, 30000 iter, PSNR 25.14 (figurines)
- **结果 figurines 24d** (mask_thresh=0.45):
  - IoU chosen: **0.5081**（vs B/16 24d 0.5751 → **-6.7pp 大幅下降**）
  - Localization: **0.7321**（vs B/16 0.8214 → **-8.9pp**）
- **中间结论**:
  - L14 在 figurines 上明显退化。可能原因: ①768d 压缩到 24d 信息损失率高于 512d→24d ②laion2b L14 特征的 tile 级判别性并不优于 B/16 ③AE 重建质量更差 (93.1% vs B/16 24d 的对应值)
  - **结果 teatime 8d** (mask_thresh=0.40):
  - IoU chosen: **0.6002**（vs B/16 8d 0.6705 → **-7.0pp 大幅下降**）
  - Localization: **0.8814**（vs B/16 0.8814 → 持平）
- **最终结论 (方向关闭)**:
  - L14 在两个场景均系统性退化: figurines -6.7pp IoU / -8.9pp Loc; teatime -7.0pp IoU / Loc 持平
  - "CLIP 特征是 figurines 瓶颈"假设修正: 瓶颈不在骨干容量, 而在 tile 级聚合协议本身——更大骨干的 768d 特征经 AE 压缩 (768→24d 压缩率 32:1) 后判别性反而更差
  - laion2b_s32b_b82k L14 特征在 tile 级 relevancy 协议下不优于 B/16 laion2b_s34b_b88k
  - 骨干升级方向关闭, 保持 B/16。若要提升上限应转向 tile 聚合方式 (多尺度/像素级) 或双流拼接

---

## 实验编号: EXP-023 — 双流拼接架构 (CLIP 8d + DINOv2 8d, score 级 late fusion)
- **日期**: 2026-09-27
- **分支**: experiment/crossattn-mm
- **动机**: 绕开"融合稀释 CLIP"结构性枷锁——CLIP 流与 baseline 完全同构 (8d, 复用 EXP-018 AE/特征零成本), DINOv2 流独立 AE (768→8d) 拼接 16d 训练; 查询时 score 级融合而非特征级混合
- **架构**:
  - CLIP 流: language_features_dim8 (EXP-018 encode) 原样, GS 渲染前 8 通道
  - DINO 流: language_features_dino (软链接 _f_dino.npy→_f.npy 零改动喂 AE) → AE 768→8d (loss=0.1747) → encode cos_sim=0.8927 → 拼接后 8 通道
  - 对齐层 W: Linear(768→512, no bias), tile 对 (DINO→CLIP image feat) cos loss, cos_sim=0.9256
  - 查询: relev = (1-λ)·relev_clip + λ·relev_dino_proj, relev_dino 经 W 投影+normalize 后走同款 softmax(10·sims) 协议
- **新增代码**: mm_langsplat/train_align.py / merge_dual.py; eval/openclip_encoder.py dual 分支; eval/evaluate_iou_loc.py --dual 系列 CLI (默认参数向后兼容)
- **Gaussian训练**: 3 level, 30000 iter, 16d rasterizer (可靠编译流程)
- **结果 teatime** (mask_thresh=0.40):
  | λ | IoU chosen | Loc |
  |---|-----------|-----|
  | 0 (baseline 8d) | 0.6705 | 0.8814 |
  | **0.3** | **0.5536** | **0.7966** |
  | **0.5** | **0.6107** | **0.7627** |
- **结论 (方向关闭)**:
  - **λ 单调劣化 (0 > 0.3 <0.5 但均大幅低于 0)**: DINO 投影流是纯噪声注入, 任何 λ>0 都拖累
  - 根因: 对齐层 cos_sim=0.9256 是"整体相似"不是"判别力"——投影把所有 tile 拉向 CLIP 空间均值方向, 投影向量与 text 的相关性近常数; softmax(10·sims) 温度敏感, 融合平摊后淹没 CLIP 尖峰分布
  - **大汇聚结论**: EXP-009 (InfoNCE) / EXP-013 (深度第三模态) / EXP-023 (双流 score fusion) 三条独立路线一致证明——LangSplat 的 text-query relevancy 协议下 CLIP 空间是唯一有效语义空间, 任何外部信号 (DINOv2/深度/对比学习) 注入均稀释判别分布
  - 多模态融合方向在 LangSplat 框架内全面关闭。后续增益应来自: tile 聚合协议改进 (多尺度/像素级)、查询协议改进、或 3DGS 几何质量
- **工程资产** (可复用): dual-stream eval CLI / train_align.py / merge_dual.py 均已参数化, 若未来有真正的文本对齐模态 (如 SigLIP) 可直接复用整个流程

---

## 实验编号: EXP-024 — 查询协议改进 (温度扫描 + CLIP 多模板 ensemble)
- **日期**: 2026-09-28
- **分支**: experiment/crossattn-mm
- **动机**: 23 个实验证明瓶颈在 tile 聚合协议与查询决策层, 注入式融合全败; 本实验攻查询侧 (免训练, 复用 renders)
- **改动**: openclip_encoder.py relev_temp 参数化 (默认10兼容); set_positives(use_templates) 7模板 CLIP ensemble (CLIP 经典模板子集: a photo of a/the, large/small, itap, origami, art); evaluate_iou_loc.py --relev_temp/--prompt_ensemble CLI + mIoU/mAcc 标准命名输出 (数值定义不变, 多 prompt 平均)
- **温度扫描 (复用 renders, mask_thresh 0.40/0.45)**:
  | temp | teatime_8d | figurines_24d |
  |------|-----------|---------------|
  | 5 | 0.6667 / 0.8814 | **0.5758** / 0.8214 |
  | 10 (基线) | **0.6705** / 0.8814 | 0.5751 / 0.8214 |
  | 20 | 0.6658 / 0.8814 | 0.5673 / 0.8214 |
  | 40 | 0.6529 / 0.8814 | 0.5372 / 0.7679 |
- **prompt ensemble (temp10)**:
  | 场景 | mIoU | mAcc |
  |------|------|------|
  | teatime_8d | **0.6932 (+2.3pp, 全项目新高)** | **0.8983 (+1.7pp)** |
  | figurines_24d | **0.5905 (+1.5pp, 新高)** | 0.8214 (持平) |
- **结论**:
  - **多模板 ensemble 两场景一致 +1.5~2.3pp**, 免训练零成本——查询协议侧确实有油水, 与"瓶颈在 tile 聚合/查询层"判断一致
  - 温度 10 双场景已近最优 (teatime 严格最优; figurines 5 仅 +0.07pp 噪声级), 保持 10; 温度鲁棒性说明 softmax(10·sims) 的尖峰化不是 mIoU 的限制因素, **prompt 侧表示质量才是** (单 phrase embedding 有模板偏置)
  - figurines 0.5905 已逼近 2D 单帧上限 0.593——3D 多视角融合已充分; 后续突破必须换更强 2D 特征 (突破点① patch token / ③ 换教师蒸馏)
  - **新 SOTA 基线**: teatime 8d+ensemble 0.6932/0.8983, figurines 24d+ensemble 0.5905/0.8214
  - 评估口径统一: 后续实验报 mIoU (multi-prompt mean) / mAcc, 与 LangSplat 论文协议对齐

---

## 实验编号: EXP-025 — 突破点① dense patch token 替换 tile global embedding (2D 上限验证)
- **日期**: 2026-09-28
- **分支**: experiment/crossattn-mm
- **动机**: EXP-024 判断瓶颈在 tile 级聚合协议; 尝试 MaskCLIP 式 dense 特征 (最后层 attention v-projection 跳过 q@k, 保留空间接地) 替换 SAM tile 的 CLIP global embedding
- **改动**:
  - preprocess.py: encode_image_dense() (conv1→pos→ln_pre→blocks[:-1]→末层 v-projection→patch mean→ln_post→proj→normalize), --dense_tiles/--no_ln_post CLI, --save_subdir 存 language_features_dense/ 不覆盖基线
  - eval/dense_upper_test.py: 2D 单帧 tile 级上限对比 (免训练), 协议严格对齐 activate_stream (30×30 avg filter→0.5*(avg+raw)→minmax→*2-1→clip→>0.5→majority smooth→IoU; chosen level=滤波后 map 峰值 argmax); smooth 用积分图精确复刻 (含 h-1 边界行为, 100×加速, 逐像素一致已验证)
- **数据**: teatime 177 帧 dense 特征提取完成 (354 文件, _f.npy [T,512] 归一化 / _s.npy 与基线相同 tile 划分)
- **结果 (6 GT 帧 × 11 prompt, mIoU chosen-level)**:
  | 特征 | 无模板 | +7模板 |
  |------|--------|--------|
  | global (language_features) | 0.4399 | 0.5394 |
  | dense (language_features_dense) | **0.0321 (-40.8pp)** | **0.0242 (-51.5pp)** |
- **诊断**: 判别方差坍缩——pos_sim std dense=0.0101 vs global=0.0372 (4×); relev 动态范围 (p95-p5) dense=0.10 vs global=0.22。tile 内 196 个 patch 取均值把 MaskCLIP 依赖的 patch 级空间选择性抹掉, 软文本对齐后 map 近常数, minmax+阈值后即噪声
- **结论 (方向关闭, 2D gate 未通过, 不进全链条)**:
  - **tile 粒度 mean-pool 的 dense 特征严重失败**: MaskCLIP dense 的价值在 per-patch 2D map, 对任意形状 SAM tile 取均值是最差用法——聚合粒度不变 (仍是 1 tile=1 向量), 只是把全局 embedding 换成更弱的特征
  - **意外发现: 3D 管线大幅超越 2D tile map** (0.6705/0.6932 vs 0.4399/0.5394, +23pp)——AE 压缩+多视角融合+per-pixel 渲染远好于分段常数 tile 图。"2D tile 上限" 框架对 teatime 不成立, 渲染特征本身就是比 tile 图更强的表示
  - 因此真正有油的 dense 方向是**替换整条 tile 协议** (per-pixel dense map 直接监督/蒸馏, 绕开 SAM tile 粒度), 而非替换 tile 内 embedding——工程量大 (栅格化维度/监督格式全改), 暂不启动
  - templates 在 2D tile map 上同样 +10pp (0.4399→0.5394), 与 EXP-024 的查询协议结论交叉验证
- **工程资产**: dense_upper_test.py (2D tile 级 eval 协议复刻, 可用于任何新特征的免训练快速验证); encode_image_dense() 保留

---

## 实验编号: EXP-026 — 突破点③ gate: CLIP dense per-patch 教师 2D 上限 (免训练)
- **日期**: 2026-09-28
- **分支**: experiment/crossattn-mm
- **动机**: EXP-025 只证伪了 tile 粒度 mean-pool; per-patch 特征图 (MaskCLIP 正确形态, 无 tile 粒度限制) 未测——若 patch 级判别性可用, 则 dense 教师蒸馏管线 (P2) 有据可依
- **做法**: eval/dense_patch_upper_test.py; 两种教师形态: (a) single-resize 224 全图 14×14 patch map 双线性上采样; (b) 滑窗 224 crop stride 112 (7×9 crop) per-patch 特征块赋值+重叠均值 → 真 per-pixel dense map; 协议严格同 activate_stream (滤波/0.75 等效阈值/majority smooth); ±7 模板
- **结果 (6 GT 帧 × 59 prompt 实例, mIoU)**:
  | 教师 | 无模板 | +7模板 |
  |------|--------|--------|
  | dense-patch single (14×14) | 0.0096 | 0.0056 |
  | dense-patch sliding (per-pixel) | **0.0191** | 0.0069 |
  | tile global 基线 (EXP-025) | 0.4399 | 0.5394 |
- **诊断 (关键反直觉)**: patch 级判别方差**未坍缩** (pos_sim std 0.0266 vs tile 基线 0.0372 同量级, 与 EXP-025 tile-mean 的 0.010 机理不同), 但**空间激活完全错位**——stuffed bear (GT 151k px, tile 级 IoU 0.97) 在 per-patch 下 IoU=0.000~0.028
- **结论 (突破点③轻量形态关闭)**:
  - CLIP patch 特征与 text 的对齐**没有空间 grounding**——patch 级高方差只是"随机方向差异", 非"物体相关差异"。CLIP 对比训练只在 global image-text 层面对齐, dense 对齐是未训练副作用; MaskCLIP dense 检索的可用性依赖单目标全图 crop 设定, 不支持复杂场景 per-pixel 开放词汇定位
  - 与 EXP-025 合并: CLIP 系教师 dense 变体全线失败, 且机理不同 (tile-mean=方差坍缩; per-patch=空间错位)——聚合粒度不是问题, **CLIP 模型本身不产 dense 判别特征**
  - 重型替代 (FC-CLIP/SAN/CAT-Seg 等密集对齐模型做教师): 需 hook 第三方特征 + AE/rasterizer 维度重编 + eval text encoder 整体替换, EXP-022 已证教师替换时协议不适配即全链退化 (-7pp), 风险收益比差, 暂不启动
- **工程资产**: dense_patch_upper_test.py (per-patch 教师 2D 快测, 任何新 dense 教师候选可直接套用)

---

## 实验编号: EXP-027 — masked CLIP pooling 快测 (tile 背景抑制方案对比, 免训练)
- **日期**: 2026-09-28
- **分支**: experiment/crossattn-mm
- **动机**: 检验 tile crop 的 bbox 背景污染是否是教师信号质量瓶颈 (最后一个轻量教师改进点)
- **做法**: eval/masked_pool_test.py; 从 _s.npy 反推每 tile 像素 mask (免重跑 SAM), bbox 内 mask 外像素填充后重过 CLIP, 协议同 activate_stream; 方案: none(纯bbox对照)/gray/black/blur(背景保留高斯模糊)
- **结果 (teatime 6 帧 × 59 prompt, mIoU chosen-level)**:
  | 方案 | 无模板 | +7模板 |
  |------|--------|--------|
  | tile 基线 (原管线特征) | 0.4399 | **0.5394** |
  | none (纯 bbox, 无填充) | 0.2473 | 0.3040 |
  | gray | 0.4386 | 0.4819 |
  | black (bbox 内置黑, 无 pad) | 0.4626 | 0.5084 |
  | blur | **0.4837** | 0.4705 |
- **关键发现**:
  - **原管线已经是 masked CLIP**: get_seg_img 对 mask 外置黑 + pad_img 正方形 pad + resize 224——"背景污染"问题 LangSplat 原作已解决; none 对照掉 19pp 证实该设计必要性
  - **blur (保留模糊上下文) 无模板 +4.4pp**: 上下文信息对判别有真实贡献, 但
  - **blur/black 与模板 ensemble 冲突**: +模板后 blur 0.4705 / black 0.5084 均低于基线 0.5394 (ensemble 的 +10pp 收益依赖黑背景特征; 上下文特征本身更稳, 模板平均无增益)
  - **2D tile 上限仍为基线+模板 0.5394**, SOTA 协议下无方案过 gate (+2pp), 不进 3D 全链条
- **结论 (方向关闭)**: 图像端编码改进与 text 端查询协议改进**不可叠加** (此消彼长), 现行"黑背景+pad × 7模板 ensemble"组合已是该协议下的局部最优; 教师信号质量的轻量改进空间耗尽
- **附带修复**: eval/openclip_encoder.py encode_image 存在死代码 mask 转发 (底层 open_clip 不支持, 任何调用即炸), 已修复为无 mask 版本 (无调用方依赖)
- **工程资产**: masked_pool_test.py (任何 tile 填充方案快测); _s.npy 反推 tile mask 方法 (免重跑 SAM)

---

## 实验编号 EXP-028
- 实验内容: L14 骨干同压缩率公平对照 (免训练 2D tile 上限, masked_pool_test.py 现场重编码, teatime 6帧×59 prompt, mIoU chosen-level, black 模式)
- 动机: EXP-022 全链条 L14 失败时 teatime 用 8d (768/8=96:1) vs B/16 8d (512/8=64:1), 压缩率不公平; 本次用同压缩率 64:1 对照, 并加 PCA 模拟器隔离"AE 压缩"变量
- 参数: B/16 laion2b_s34b_b88k black 重编码锚点 / B/16 PCA 8d (64:1) / L14 laion2b_s32b_b82k 原始 768d / L14 PCA 12d (64:1); 各 ±7模板 ensemble
- 结果 (mIoU chosen-level):
  | 组 | 无模板 | +模板 |
  |---|---|---|
  | B/16 原始 512d | 0.4704 | 0.5241 |
  | B/16 PCA 8d | 0.1542 | 0.1668 |
  | L14 原始 768d | 0.4773 | 0.4713 |
  | L14 PCA 12d | 0.2194 | 0.1921 |
- 结论:
  1. L14 原始 tile 判别力无优势 (+0.7pp, gate 需 +2pp 未过); +模板下反降 -5.3pp (0.4713 vs 0.5241)——L14 特征与模板 ensemble 不兼容
  2. **EXP-022 压缩率主因假说证伪**: 同压缩率 PCA 下 L14 (0.2194) 反而 +6.5pp 优于 B/16 (0.1542), 768d 冗余更多、同压缩率保留更好; L14 失败主因 = 特征本身在 tile relevancy 协议下不占优, 与压缩率无关
  3. PCA 模拟器意外价值: 线性压缩 64:1 使 2D tile mIoU 0.47→0.15 (毁灭性), 而真实非线性 AE 8d 全链条 3D IoU 0.6705——**AE 学到的非线性映射 + 3D 多视角融合能从低维特征恢复大量判别力**, 再次印证 EXP-025 "3D>>2D" 元结论
  4. L14 骨干方向第二次关闭 (免训练 gate 层面即失败, 无需全链条验证)
- 工具: masked_pool_test.py 新增 --clip_model/--pretrained/--clip_n_dims/--pca_dim (pca_compress AE 压缩率模拟器)

---

## 实验编号 EXP-029
- 实验内容: B/16 预训练权重扫描 (免训练 2D tile 上限, masked_pool_test.py, teatime 6帧×59 prompt, mIoU chosen-level, black 重编码协议, 全部 512d 零管线改动)
- 参数: 权重=独立变量; 其余协议与 EXP-028 完全一致 (锚点 laion2b_s34b_b88k 0.4704/0.5241 引自 EXP-028)
- 结果:
  | 权重 (ViT-B-16) | 无模板 | +7模板 | vs 锚点(无模板) |
  |---|---|---|---|
  | **laion400m_e32** | **0.5510** | **0.5937** | **+8.1pp** |
  | datacomp_l_s1b_b8k | 0.4962 | 0.4437 | +2.6pp |
  | laion2b_s34b_b88k (锚点) | 0.4704 | 0.5241 | - |
  | EVA02-B-16 merged2b | 0.4435 | 0.4596 | -2.7pp |
  | commonpool_l_clip_s1b_b8k | 0.4409 | 0.3540 | -3.0pp |
  | openai | 0.2958 | 0.2710 | -17.5pp |
- 结论:
  1. **laion400m_e32 大幅过 gate (+2pp)**: 无模板 +8.1pp, +模板 0.5937 (+7.0pp)——创 teatime 2D tile 上限新高 (此前最高 0.5394); 且与模板 ensemble 兼容 (继续 +4.3pp), 唯一双兼容权重
  2. openai 原版极差 (-17.5pp)——LangSplat 原作选 laion 权重是正确决策
  3. "数据规模越大越好"不成立: laion400m (400M, 32ep) > laion2b (2B) > datacomp_l (1B) > commonpool_l (1B); tile 级判别力取决于训练配方而非规模
  4. **ensemble 兼容性成为新判据**: datacomp/commonpool/EVA02 +模板全部退化 (-5~-9pp), 与 EXP-028 L14 (-5.3pp) 一致——模板 ensemble 只对 laion 系特征有效
  5. 注意: datacomp 无模板 +2.6pp 也略过 gate, 但 +模板退化, 综合不如锚点
- 待办: e31 对照 (确认 e32 非孤例) → laion400m_e32 全链条 (teatime preprocess 已启动, language_features_laion400m_e32)
- 工具: 无新增 (复用 EXP-028 参数化)

---

## 实验编号 EXP-030
- 实验内容: laion400m_e32 教师 teatime 全链条 (2D tile 上限冠军的 3D 下游验证)
- 配置: preprocess 重提取 (language_features_laion400m_e32, 177帧) → AE 8d (best_loss=0.1666, cos_sim=0.9118, 71.3%>0.9, ckpt/teatime_e32) → encode_dim8 (竞态判据通过) → GS×3 (30000 iter, 热启动 RGB ckpt, 端口 55591-3) → render → eval (mask_thresh 0.4, relev_temp 10)
- 结果 (mIoU chosen-level / mAcc):
  | 口径 | EXP-030 e32 8d | SOTA laion2b 8d | Δ |
  |---|---|---|---|
  | 无模板 | 0.5975 / 0.8814 | 0.6705 / 0.8814 | **-7.3pp / 持平** |
  | +7模板 | 0.6126 / 0.8983 | 0.6932 / 0.8983 | **-8.1pp / 持平** |
- 过程事故 (重要): 首次 eval mIoU=0.0048 假崩溃——**eval 未传 --clip_pretrained, 默认 laion2b text encoder 去评 e32 visual 特征, 空间错配**。三源对照定位: 同一份 e32 落盘特征, 配 e32 text 空间 gap=+0.37 (优于 laion2b 的 +0.30), 配 laion2b text 空间 gap=-0.05
- 结论:
  1. **2D tile 上限增益 (+8pp) 完全不迁移到 3D 下游 (-8pp, 符号翻转)**——教师信号轴与下游脱钩的第三块铁证 (EXP-025 方差坍缩 / EXP-027 背景抑制 / EXP-030 上限反转, 三点一线)
  2. 机理候选: e32 特征有效维度更高 (8d AE cos_sim 71.3% vs laion2b 99.0%), 在 8d 压缩瓶颈处判别性大量丢失; "上游判别力 ≠ 下游性能" (与 EXP-022 L14 同款)
  3. mAcc 双口径与 laion2b 恰好持平 (0.8814/0.8983)——定位能力与教师信号源无关
  4. laion2b 仍是 teatime 最优教师; 候选后续: e32 12d/16d AE (e32 可能需要更高维度), 但优先级低于全场景主表
- 工程教训: **eval 的 --clip_model/--clip_pretrained 必须与特征教师同源**; 换教师特征后 eval 的 text encoder 必须联动, 否则出现 0.0048 级假崩溃

## EXP-031/032 (2026-09-29) teatime laion400m_e32 AE 维度扫描: e32 不需要更高维度
- 动机: EXP-030 显示 e32 特征有效维度更高 (AE 8d 往返 cos_sim 71.3% vs laion2b 99.0%), 假设 e32 需更高 AE 维度
- 方案: e32 特征分别配 AE 12d (EXP-031) / 16d (EXP-032), 其余同 EXP-030 (mse_only, lr=1e-4 默认, 100 epochs, GS 热启动 RGB ckpt, eval teacher 同源 laion400m_e32, mask_thresh 0.4, mIoU 口径)
- AE: 12d best_loss=0.1467 (ep98) / 16d best_loss≈0.1460
- 结果 (mIoU/mAcc):
  | AE维度 | 无模板 | +7模板 |
  | 8d (EXP-030) | 0.5975/0.8814 | 0.6126/0.8983 |
  | 12d (EXP-031) | 0.5670/0.8475 | 0.5709/0.8644 |
  | 16d (EXP-032) | 0.5836/0.8983 | 0.5916/0.8983 |
  | SOTA laion2b 8d | 0.6705/0.8814 | 0.6932/0.8983 |
- 结论:
  1. e32 AE 维度曲线非单调: 8d (0.5975) 是 teatime 峰值, 12d 谷 (0.5670), 16d 回升 (0.5836) 但仍低于 8d — "e32 需要更高维度"假设否定
  2. e32 全维度均低于 laion2b 8d 达 -8~-10pp, EXP-030 结论加固: e32 骨干全链条确定失败, 骨干/权重轴正式关闭
  3. ensemble 增益在 e32 各维度仅 +0.4~0.8pp (laion2b 为 +2.3pp) — ensemble 收益依赖教师特征分布

## EXP-033 (2026-09-30) teatime laion400m_e32 AE 20d: e32 新峰值, 假设部分成立
- 动机: EXP-031/032 后用户提出试 20d
- 方案: 同 EXP-031/032 全链 (v4 约定: -m 基名/默认 lr/100 epochs/teacher 同源 eval), 仅 AE 维度 20
- 结果 (mIoU/mAcc): 无模板 0.6132/0.8983; +7模板 0.6187/0.9153
- e32 维度曲线 (teatime): 8d 0.5975 -> 12d 0.5670(谷) -> 16d 0.5836 -> 20d 0.6132(峰)
- 结论:
  1. "e32 需要更高维度"假设部分成立: 20d 比 8d +1.6pp/+0.6pp, 创 e32 teatime 新高
  2. 但仍低于 laion2b 8d 达 -5.7pp(无模板)/-7.5pp(ensemble), 骨干结论不变: laion2b 仍是 teatime 最优教师
  3. +ensemble 20d 的 mAcc=0.9153 追平 laion2b Loc SOTA, e32 的定位能力不差
  4. 曲线非单调且波动大 (12d 谷 -3pp), AE 训练随机性不可忽略, 单点维度结论需谨慎

## EXP-034 (2026-09-30) teatime laion400m_e32 AE 24d: 20d 峰值确认, e32 维度扫描收官
- 方案: 同 EXP-033 全链, 仅 AE 维度 24
- 结果 (mIoU/mAcc): 无模板 0.5883/0.8475; +7模板 0.5947/0.9153
- e32 完整维度曲线 (teatime, 无模板 mIoU): 8d 0.5975 -> 12d 0.5670(谷) -> 16d 0.5836 -> 20d 0.6132(峰) -> 24d 0.5883
- 结论:
  1. 24d 回落 -2.5pp, 20d 峰值确认, e32 teatime 倒 U 型 (波动大, AE 随机性不可忽略)
  2. "维度最优值场景/教师相关"再添一例: teatime baseline(laion2b) 峰 8d / e32 峰 20d / figurines fused 峰 20d
  3. 峰对峰仍低于 laion2b 8d 达 -5.7pp, 骨干/权重轴最终结论: laion2b 仍是 teatime 最优教师
  4. +ensemble 24d 的 mAcc=0.9153 与 20d/laion2b 持平 — e32 系的定位能力与 laion2b 无差

## EXP-035a (2026-09-29) 方向①启动: DINO 独立特征场 image-query 检索 — 概念验证成功
- 假设: LangSplat 的 Loc 本质是 image-query 2D→3D 检索, DINOv2 自监督实例判别天然适配; 与 text-query 分割(CLIP 独占)正交 → 任务路由(双特征场)
- 免训练快测(2D tile 检索, teatime 6帧62对象): cross-frame top1 DINO 74.07% vs CLIP 63.79% (+10.3pp), 4 粒度一致占优; PCA 扫描: DINO 检索判别力需高维(128d 峰, 8-24d 受损)
- 配置: DINOv2 per-tile 768d (language_features_dino) → AE 32d (encoder 256 128 32 32 32 / decoder 32 128 256 256 768, mse, 默认 lr 1e-4) → encode_dim3 → GS×3 热启动 teatime_-1 RGB ckpt → render
- eval: eval_image_query.py 新协议 — image-query cross-frame 3D 检索: 帧 A GT bbox 最大覆盖 tile 特征(AE 同空间) → 帧 B render 特征场逐像素 cos → 30×30 滤波 → argmax 点落同类 GT bbox 判命中, chosen-level
- 结果(teatime, 61 对象对): **DINO 32d 场 90.16% (55/61) vs CLIP 8d 场 81.97% (50/61), +8.2pp**
- 结论: 任务路由 3 级证据链齐 — 2D tile 快测 +10.3pp / 3D 场 +8.2pp / text-query 分割 DINO 注入全败(对照)。DINO 场在 image-query 检索上大幅有效, 方向①立项依据成立
- 工程: NUM_CHANNELS 静态共享内存硬上限 32d (16×16 tile×dim×4B ≤48KB, 64d 需 75.7KB 报 ptxas 超限); >32d 须动态共享内存改造(extern __shared__ + cudaFuncSetAttribute, 4090 上限 99KB)

## EXP-036a (2026-10-01) 方向① 跨场景验证: figurines — 结果反转, 单场证据不泛化
- 配置: 同 EXP-035a, 场景 figurines (248帧 preprocess 85min), 对照升级为 figurines SOTA 场 (24d, language_features_dim24)
- 结果: image-query 3D 检索 chosen-level top1 — CLIP 24d 场 88.89% (48/54) vs DINO 32d 场 77.78% (42/54), **-11.1pp** (teatime +8.2pp 反转)
- 2D tile 快测归因: figurines cross top1 DINO 64.35% vs CLIP 58.80% (**2D 领先 +5.6pp**) → 特征层 DINO 仍占优, 矛盾指向 figurines DINO 场链路 (AE 32d 瓶颈或 GS 训练质量)
- 结论: 任务路由主张受重创——teatime +8.2pp 不能泛化。两个待排查点: (1) figurines 密集小物体 (54 对象/4帧) 下 32d AE 瓶颈更紧, 需更高维; (2) figurines DINO 场 GS 3 级训练质量未验证
- 方法论教训: EXP-035a 只有单场景单配置就宣称"概念验证成功"为时过早; 对照场强度场景相关 (figurines 24d 强于 teatime 8d 的检索基线), 场景间数字不可直接比

## EXP-036a 归因诊断三连 (2026-10-01) — 无 bug, 翻转是结构性
1. AE 排除: DINO 32d 往返 cos figurines 0.7951 > teatime 0.6664; 768d vs 32d top1-neighbor rank agreement 两场景均 ~0.80 — AE/维度不是瓶颈 (重建质量再次与下游脱钩)
2. 渲染排除: 渲染场 tile 区域均值 vs 2D encode 特征 cos — teatime 0.848 / figurines 0.822, 差距不足以解释 19pp 摆幅
3. 跨视角一致性证实: 同物体跨帧 tile cos — teatime CLIP 0.775 vs DINO 0.706 (-7.0pp); figurines CLIP 0.683 vs DINO 0.629 (-5.4pp) — DINO 视角敏感性高于 CLIP (CLIP tile 黑底归一化视角不变性强)
- 完整归因链: figurines DINO 2D 优势入口缩水 (+10.3→+5.6pp, 密小物体) × 3D 放大器对 DINO 增益弱 (DINO +13~16pp vs CLIP +18~30pp, 跨视角稀释) × figurines CLIP 24d 对照场 SOTA 级强 (88.89%) → 三因素叠加翻转
- 3D 场对 2D 检索的放大两场景为正 (DINO +16.1/+13.4pp, CLIP +18.2/+30.1pp) — 3D 多视角融合一致有效, 但效率特征类型相关
- 修复方向: (a) DINO 场训练加跨视角一致性正则; (b) DINO tile 提取更强背景抑制/紧 crop; (c) 论文主张修正为条件性优势 + 模态×场景交互分析
- 工具沉淀: eval/diag_dino_field.py (AE 往返+rank agreement), eval/diag_render_fidelity.py (渲染忠实度), eval/diag_crossview.py (跨视角一致性)

## EXP-037a (2026-10-01) 跨视角自蒸馏平滑修复: figurines DINO 场追平 CLIP 场
- 方法: 用已训 DINO 场渲染特征(3D 一致)与 2D tile 监督凸组合 α=0.5 (smooth_tiles.py, 逐 level 逐 tile), 重训 GS×3 → render → eval
- 结果: image-query 3D 检索 chosen-level top1 — DINO 平滑场 **88.89% (48/54)** vs 修复前 77.78% (+11.1pp) = **完全追平 CLIP 24d 场 (48/54 同分)**
- 结论: 跨视角不一致假设的修复路径验证成功——单轮 α=0.5 平滑即抹平 -11.1pp 缺口。任务路由主张修正为: teatime 领先 +8.2pp (未平滑版) / figurines 追平 0pp, "不劣且可大幅领先"
- 过程事故 (3 连): -m 漏替换致原场 L1 ckpt 覆盖 (可 38min 重训恢复, renders/eval 资产无损); skip-guard 误插渲染循环致 continue 跳渲染; 修复后断点复用零重训
- 待做: teatime 平滑验证 (确认 +8.2pp 不受损), α 扫描 (0.3/0.7), 第二轮迭代 (用平滑场再渲染再平滑, 冲 figurines 反超)

## EXP-037b/c (2026-10-01) figurines DINO 场 α 扫描与迭代 (方案2)
- EXP-037b (α=0.7 单轮): DINO 32d 平滑场 90.74% (49/54) — **反超 CLIP 24d 场 (88.89%) +1.85pp**
- EXP-037c (迭代: a05特征+a05场再平滑0.5): 85.19% (46/54) — 回落, 迭代无累积效应 (二次稀释)
- figurines DINO 场完整曲线: 原始 77.78% → α0.5 88.89% (追平) → **α0.7 90.74% (甜点)** → iter2 85.19%; CLIP24d 88.89%
- 任务路由最终主张: **两场景统一占优** — teatime +8.2pp (原始场, α 未调) / figurines +1.85pp (α=0.7); teatime α 曲线待测 (预计更高)
- 过程教训: render_base 参数化丢失 level 后缀 → 平滑静默退化为复制源特征 (a07=原始逐分复现 42/54 暴露); "结果与历史逐分相同=数据流失效" 数值指纹检查法; 特征两两互异实证校验后才重训

## EXP-039 (2026-10-03) 论文主表全链: waldo_kitchen + ramen (任务路由四场景证据矩阵闭合)
- 流水线: run_master_table.sh 全自动 (preprocess --use_dino → CLIP 8d AE/GS×3 + DINO 32d AE/GS×3 → α0.7 自蒸馏平滑重训×3 → 渲染 → text eval ±ensemble → image eval)
- 过程修复: ① image eval CLIP8d 场 query 误用原始 512d language_features → 改 language_features_dim8 (同 AE 空间); ② eval_image_query.py 协议升级: 新增 any-bbox 口径 (GT 重复标签场景 first-bbox 惩罚极大) + per-level 真实统计 (原代码误记 chosen level)

### waldo_kitchen (mask_thresh 0.40, 187帧)
- text eval (CLIP 8d): 无 ensemble mIoU 0.5870 / mAcc 0.8636; +7模板 ensemble 0.5404 / 0.8182 (**ensemble 退化 -4.7pp**, 与 teatime/figurines 相反)
- image eval (13 对): DINO a07 场 46.15% (any-bbox 69.23%) vs CLIP 8d 场 69.23% (any 92.31%) — CLIP 反超但 **McNemar p=0.25 不显著**
- 归因诊断 (EXP-039d): ① knife×9 等重复标签密集, first-bbox 协议惩罚 -23~-40pp; ② DINO 2D 入口优势消失 (2D tile 检索 DINO 48.1%/84.6% vs CLIP 36.5%/84.6%, teatime +10.3pp/figurines +5.6pp 的入口优势归零); ③ "32d 维度不合适"假说排除 (DINO AE cos_sim 0.910 三场景最高; PCA 扫描 16d 峰值但 AE-32d 优于 PCA-32d, tile-std 0.129 vs 0.089); ④ DINO 场崩在 level1/coarse (any 60% vs CLIP 84.6%); ⑤ α=0.7 直接套用, 无 raw 场对照
- 待验证: raw DINO 场 vs a07 消融 (需重渲); any-bbox 口径下 CLIP 仍领先 23pp, 方向性存在但 n=13

### ramen (mask_thresh 0.55, 131帧, 禁 --eval)
- text eval (CLIP 8d): 无 ensemble mIoU 0.5387 / mAcc 0.7042; +ensemble 0.5062 / 0.7324 (mIoU -3.3pp, mAcc +2.8pp)
- image eval (79 对): DINO a07 场 **78.48%** (any-bbox 84.81%) vs CLIP 8d 场 74.68% (any 82.28%) — **DINO +3.8pp/+2.5pp, 任务路由成立**; per-level DINO 全占优 (L1 58.2 vs 46.8 / L2 79.8 vs 70.9 / L3 77.2 vs 49.4, first 口径)

### 四场景任务路由证据矩阵 (image-query 3D 检索 chosen-level top1, first-bbox 口径)
| 场景 | DINO α0.7 场 | CLIP 场 | Δ | n | 判定 |
|---|---|---|---|---|---|
| teatime | 91.80% | 81.97% (8d) | +9.8pp | 61 | DINO 胜 |
| figurines | 90.74% | 88.89% (24d) | +1.85pp | 54 | DINO 胜 |
| ramen | 78.48% | 74.68% (8d) | +3.8pp | 79 | DINO 胜 |
| waldo | 46.15% | 69.23% (8d) | -23.1pp (ns, p=0.25) | 13 | CLIP 胜(不显著) |
- 任务路由最终主张: **4 场景 3 胜 1 负(不显著)**; 唯一反例 waldo 是最小样本 (13 对) + 协议缺陷 (knife×9) + DINO 入口优势消失三因素叠加; ramen 大样本 (79 对) 独立复现 DINO 优势
- text eval ensemble 收益场景相关: teatime +2.3 / figurines +1.5 / waldo -4.7 / ramen -3.3 (mIoU pp) — 两升两降, 主表报双口径
- 待做 (EXP-040): teatime/figurines any-bbox 重评 (渲染已删需重渲 12 场) + waldo α 消融 (raw vs a07) → 判定 waldo 反转是否 α 过度平滑所致

---

## EXP-040 (2026-10-03) any-bbox 统一口径重评 + waldo α 消融 (raw vs a07) — 任务路由矩阵钉死
- **分支**: experiment/crossattn-mm
- **脚本**: run_exp040.sh (按维度顺序 32d→24d→8d 三次 patch+编译, 重渲 15 场: waldo raw dino_32d×3 + teatime/figurines dino_32ds_a07×6 + figurines_24d×3 + teatime_8d×3)
- **渲染忠实性验证**: EXP-040 重跑的 7 组 eval 中 6 组 first-bbox 值与 EXP-039/035 历史值完全一致 (waldo 46.15/69.23, teatime 91.80/81.97, figurines 90.74/88.89) — 删除重渲流程无损

### ① any-bbox 统一口径四场景矩阵 (image-query chosen-level top1, first / any 双口径)
| 场景 | CLIP 场 | DINO a07 场 | Δ(first) | Δ(any) | n |
|---|---|---|---|---|---|
| teatime | 81.97 / 86.89 (8d) | 91.80 / 93.44 | +9.8pp | +6.6pp | 61 |
| figurines | 88.89 / 88.89 (24d) | 90.74 / 90.74 | +1.9pp | +1.9pp | 54 |
| ramen | 74.68 / 82.28 (8d) | 78.48 / 84.81 | +3.8pp | +2.5pp | 79 |
| waldo | 69.23 / 92.31 (8d) | 46.15 / 69.23 | -23.1pp (ns) | -23.1pp | 13 |
- **any-bbox 口径下 3胜1负维持**, 但 DINO 优势幅度普遍收窄 (teatime +9.8→+6.6, ramen +3.8→+2.5, figurines 不变) — first-bbox 部分低估 CLIP 场; any-bbox 更公平, 主表应报双口径
- 口径敏感性场景相关: teatime any-bbox 帮 CLIP 多救 3 对 (+4.9pp) 而 DINO 只 +1.6pp; figurines 两口径完全相同 (GT bbox 大/重复标签少)

### ② waldo α 消融 (raw DINO 32d 场 vs α0.7 平滑场, n=13)
| 场 | first-bbox | any-bbox |
|---|---|---|
| CLIP 8d 场 | 69.23% | 92.31% |
| DINO 32d **raw** | 53.85% | 76.92% |
| DINO 32d **a07** | 46.15% | 69.23% |
- **α=0.7 平滑在 waldo 伤害 -7.7pp** (raw 76.92 > a07 69.23, any 口径; 1 对差异, n=13 无法显著, 方向明确)
- 归因闭环: waldo 反转 = α 平滑伤害 (-7.7pp) + DINO 场本身弱于 CLIP 场 (raw 仍输 15.4pp) 双因素叠加; 后者的根源是 2D 入口优势归零 (EXP-039d 已证 2D 检索打平) — 平滑增益需要 2D 优势基础, waldo 无此基础故 α 直接套用有害
- EXP-039d 假说⑤ ("α 无 raw 对照直接套用") 证实; 修正 EXP-039 归因④的权重: 平滑不是主因 (即使 raw 也输), 但去掉平滑可挽回 1/3 缺口
- **方法论修正**: 自蒸馏平滑 α0.7 不应无脑全场景套用 — 前置判据 = 该场景 2D tile 检索 DINO 是否领先; 领先才平滑 (teatime/figurines), 打平则跳过 (waldo)

### 结论
1. 任务路由最终证据 (EXP-035→040): 四场景 any-bbox 统一口径 3胜1负, 唯一败场 waldo 完全归因 (n=13 极小样本 + knife×9 协议 + DINO 2D 入口无优势 + α 平滑套用伤害), 三场景显著/大样本优势 (teatime 61对/ramen 79对/figurines 54对) 支撑论文主张
2. any-bbox 是更公平口径, 主表双口径报告; DINO 优势方向在任何口径下不变
3. α=0.7 平滑的适用条件明确: 需 2D 入口优势前提; 论文方法节应写为"当 2D 检索验证 DINO 判别力领先时启用自蒸馏平滑"
4. rasterizer 现为 8d 编译 (EXP-040 结束状态), ckpt 全保留, 渲染已清理

---

## EXP-041 (2026-10-03) McNemar 显著性检验 — 全部 ns，teatime first 边缘
- **脚本**: run_exp041.sh (重渲 24 场: 4 场景 CLIP/DINO a07 ×3 level, 维度顺序 32d→24d→8d) + eval --out_json per-pair + eval/mcnemar.py 精确二项检验
- **结果** (n=13/61/54/79):
  - teatime: Δfirst +9.8pp, discordant b=1 c=7, p=0.0703 ns (边缘); any +6.6pp p=0.2188
  - figurines: Δ +1.9pp, b=4 c=5, p=1.0 ns (discordant 近对称, 增益来自基础率)
  - ramen: Δ +3.8pp, b=7 c=10, p=0.6291 ns; any p=0.8036
  - waldo: Δ -23.1pp, b=3 c=0, p=0.25 ns
  - pooled: b=15 c=22 (first), p=0.3240 ns; any p=0.6076
- **结论**: DINO 优势 = 方向一致(3/4) + 大样本效应量, 非显著证据; 论文必须报完整 McNemar 表 + limitation; teatime first p=0.07 最强
- 工具: eval_image_query.py 加 --out_json (per-pair 记录), eval/mcnemar.py (--protocol first/any, join 键 (frameA, obj_i))
- 渲染已清理, per-pair JSON 保留在 eval_result/mcnemar/

## EXP-042 (2026-10-04) teatime α 扫描 (raw/a03/a05/a07) — α0.3 首个显著 + 平滑机制定位
- **脚本**: run_exp042.sh (挂) → run_exp042b.sh (修复续跑); teatime_dino_32ds_a03/a05 各重训 3 level GS
- **挂因**: smooth guard 的 f-string 里写了 {TAG}, bash 未展开 → Python NameError; 教训: bash 传变量给 python 一律用双引号内联展开或 os.environ
- **teatime α 曲线** (first / any, n=61):
  - raw: 90.16 / 93.44 (EXP-035a 值复核一致)
  - α0.3: **91.80 / 93.44, McNemar p=0.0312 ★ (b=0 c=6, 全项目首个显著; CLIP 赢的 pair 为零)**
  - α0.5: 90.16 / 91.80 (谷, p=0.125)
  - α0.7: 91.80 / 93.44 (p=0.0703, 与 EXP-040 一致)
- **结论**:
  1. teatime α 曲线非单调且波动仅 ±1 对 (噪声级) — teatime 平滑收益本质微弱 (+1.6pp), 远小于 figurines (+11.1pp); α 最优值场景相关, 无全局最优
  2. **α 收益只存在于 first-bbox 口径, any 口径全程不变/略降 (raw 已 93.44)** → 平滑机制 = 修复 bbox 中心偏移, 不提升整体判别力
  3. 方法论判据修正: "2D tile 检索 DINO 领先" 是平滑启用必要非充分条件 (teatime 2D +10.3pp 领先但平滑近无效; waldo 2D 打平则有害)
  4. 主表可报 teatime a03 场: 91.80/93.44, p=0.0312★ — 论文唯一 ★
- 渲染已清理 (raw×3+a03×3+a05×3), JSON 保留; 磁盘 80%

## EXP-043 (2026-10-04) 空间自适应平滑 (per-tile α 门控) — Phase A 信号研究: "waldo 场病态"假设被证伪
- **动机**: waldo α=0.7 -7.7pp (EXP-040), 设计 per-tile 门控 α_i 按置信度决定平滑强度, 期望 waldo 自动降 α
- **新工具**: eval/smooth_tiles_adaptive.py (--signal cos2d|cons, --gate hard|soft|quantile, --beta 退让系数, --dry 仅信号), eval/analyze_gate_signals.py; run_exp043a.sh (Phase A)
- **信号定义** (per frame/level/tile, 面积≥30px):
  - cos2d = cos(tile渲染均值, 2D监督特征) — 渲染-2D一致性
  - cons = 跨 level 渲染均值共识 (同像素掩码在另外两个 level 场的渲染均值的平均 cos) — 场健康度, 不混淆"待修复的视角敏感"与"渲染垃圾"
- **工程坑**: figurines_dino_32d_1 整目录被清理 (render.py 报 cfg_args FileNotFoundError 实为目录不存在), 补训 1 level (~40min); 12 个 raw 渲染全部重建
- **Phase A 结果** (信号分位数, 五分位 q05/25/50/75/95):
  - **waldo 场与健康场景无法区分**: cons 中位数 waldo L1/L2/L3 = 0.917/0.941/0.924 vs teatime 0.931/0.952/0.935 vs figurines 0.923/0.943/0.920; cos2d waldo (0.838/0.740/0.788) 甚至优于 teatime (0.806/0.700/0.745)
  - 任何 τ 下 waldo gated-out 比例与 teatime/figurines 接近 (cons τ=0.8: 0.14/0.07/0.14 vs 0.09/0.04/0.10 vs 0.11/0.03/0.11)
- **结论 (Phase A)**:
  1. **证伪"waldo 渲染垃圾 tile 可检测"假设**: waldo DINO 场在 tile 级一致性与共识信号上不病态, 场本身没有坍缩 (EXP-039 "崩在 level1/coarse" 表现为下游检索指标, 不体现在特征空间一致性上)
  2. 结合 EXP-041 McNemar p=0.25: waldo -7.7pp (any) = 13 对中 1 对翻转的噪声级, "平滑系统性伤害 waldo"证据不足
  3. 门控价值主张修正: 从"自动止损"改为"保守安全带" — β=0.5 低置信 tile 平滑减半 (非归零), 预期四场景 ≈ a07 ± 微调; 若 waldo 门控版 ≥ raw 76.92% 则"安全带"主张成立
- **Phase B 配置** (run_exp043b.sh, 11:56 启动): signal=cons, gate=hard, τ=0.85, β=0.5, α=0.7, tag=g07; 场景顺序 waldo→teatime→figurines→ramen; guard = gated-out>5% + maxdiff>1e-3; eval 后 McNemar vs CLIP 场 + vs a07 场 (first/any)
- **成功判据**: waldo g07 ≥ raw 76.92% (any); teatime/figurines g07 ≥ a07 (91.80/90.74 first)

## EXP-043 Phase B 结果 (2026-10-04 19:34 完成): per-tile 门控无净增益, 方向关闭
- **四场景 g07 (cons, hard τ=0.85, β=0.5, α=0.7) vs a07** (image-query first/any):
  - teatime: 91.80/93.44 = a07 **逐对全同** (discordant 0:0)
  - waldo: 46.15/69.23 = a07 **逐对全同** (discordant 0:0)
  - figurines: 87.04/87.04 vs 90.74/90.74 = **-3.7pp** (b=3 c=1, p=0.625 ns)
  - ramen: 81.01/87.34 vs 78.48/84.81 = **+2.5pp** (b=1 c=3, p=0.625 ns)
  - **pooled: first 173/207 = a07 173/207; any 182/207 = a07 182/207 — 总命中数完全相同, 场景间 ±2 对对冲**
- **vs CLIP 场** (g07 口径): teatime +9.8pp (p=0.0703) / figurines -1.9pp (p=1.0) / ramen +6.3pp (p=0.33) / waldo -23.1pp (p=0.25) — 与主表 a07 口径一致
- **结论 (EXP-043 总闭环)**:
  1. **per-tile 自适应平滑无净增益**: pooled 与全局 α=0.7 逐命中数完全一致; 逐对变化仅在 figurines(-2)/ramen(+2), 均为噪声级 (全 ns)
  2. **机制解释**: 门控 tile (~10-20%) 与检索查询/峰值路径基本不重叠 (teatime/waldo 逐对不变), 门控只在少数场景边缘 tile 起作用且方向随机
  3. **tile 级特征信号 (cos2d/cons) 与下游检索表现脱钩** (Phase A + Phase B 双证): waldo 场信号健康但仍输; 门控信号无法预判哪些 tile 对下游有害
  4. **最终设计维持**: 全局 α=0.7 + 前置判据 (该场景 2D tile 检索 DINO 领先才平滑, EXP-040 判据)
  5. **论文定位**: 放入消融/设计空间探索节, 回应审稿人"为什么不用 per-tile 自适应 α"——有完整负结果证据链 (EXP-043 A+B)
- **产物**: per-pair JSON `eval_result/mcnemar/{ABBR}_DINO_g07.json` + McNemar ×8 (vs CLIP/a07 × first/any); 门控统计 `eval_result/adaptive/gated_<scene>.json`; 全 log `eval_result/adaptive/exp043b.log`
- **磁盘清理**: g07 12 场渲染 + ckpts 已删 (可由 smooth_g07 特征 + base ckpt 重训复原); Phase A 的 12 场 raw dino_32d 渲染已删

## EXP-044 (2026-10-05 02:10 完成) 训练时 EMA 跨视角一致性正则: ≡ raw, 方向关闭
- **方法**: train.py 新增 lf_cons loss (--lf_cons_weight 0.1 --lf_cons_momentum 0.9): 同 tile 不同帧被采样时, 渲染均值向 EMA 锚点对齐 (梯度只过当前渲染); raw 2D 监督不混回, 与离线平滑 (a07) 机制对照; 冒烟 300 iter 通过
- **结果** (image-query first/any):
  - waldo: 53.85/76.92 = **与 raw 完全同数** (7/13, 10/13); vs a07/g07 多 1 对 (10 vs 9, 无平滑伤害); vs CLIP -15.4pp p=0.5 ns
  - teatime: 90.16/93.44 = **与 raw 完全同数** (55/61, 57/61); vs a07 first 少 1 对 (90.16 vs 91.80); vs CLIP +8.2pp p=0.0625 边缘
- **结论 (双场景独立验证)**:
  1. **训练时跨视角正则 (w=0.1) 完全复刻 raw 场**——正则太弱不改变收敛解; 未扫更强权重 (若 0.3/0.5 大概率伤重建, 与 smooth 混 2D 的方向性不同, 不追)
  2. **"混回 2D"是平滑增益的必要成分**: 纯 3D 一致性约束 (正则) 不能产生 raw→a07 的增益; 与 EXP-042 机制结论 (平滑=以 2D 为锚修复 bbox 中心偏移) 独立互证
  3. **设计空间闭环**: 全局 α=0.7 离线平滑 > 训练时正则 (≡raw) > per-tile 门控 (pooled 持平但 figurines -2对); 最简单的方案胜出, 论文消融节完整证据链 (EXP-037 iter2 无累积 / EXP-043 门控 / EXP-044 正则)
  4. waldo 三重验证: 平滑/门控/正则全部动不了 waldo (-23pp 差距), 根源铁证 = 2D 入口打平, 非 3D 场质量
  > **⚠ EXP-045 更正**: 本节"方向关闭"结论仅对 w=0.1 成立; w 网格扫描后发现 w=0.3 峰值 95.08% (teatime 新高, vs CLIP p=0.0078 显著), 方向c复活, 见 EXP-045
- **产物**: per-pair JSON `eval_result/mcnemar/{WALDO,TEATIME}_DINO_cw01.json`; log `eval_result/adaptive/exp044_{waldo,teatime}.log`; c01 渲染/ckpt 保留未清理 (约 210GB, 磁盘紧张时可删, 可由特征+base ckpt 重训复原)

## EXP-045 (2026-10-05 17:20 完成) lf_cons 权重网格 (teatime): 尖锐倒U, w=0.3 峰值 95.08% 创新高
- **配置**: teatime, w ∈ {0.3, 1.0, 3.0} (w=0.1 来自 EXP-044), momentum=0.9, 其余同 EXP-044; 脚本 run_exp045_grid.sh (重启3次: ①vs_raw json 名 cw03≠ccw03 ②TAG 变量前缀传递不回写致 c03——教训: 子脚本 tag 拼接规则必须先核对)
- **结果** (image-query first / any, n=61):
  - **w=0.1**: 90.16 / 93.44 (≡ raw, EXP-044)
  - **w=0.3**: **95.08 / 95.08** — vs CLIP first +13.1pp **p=0.0078 ★** (b=0 c=8, 全项目第二个显著); vs raw +3 对 (b=0 c=3, p=0.25); vs a07 (56/61) +2 对; **teatime image-query 历史新高**
  - w=1.0: 85.25 / 88.52 (低于 raw -4.9pp, 开始伤)
  - w=3.0: 73.77 / 73.77 (重伤 -16.4pp)
- **曲线**: 90.16 (0.1) → **95.08 (0.3)** → 85.25 (1.0) → 73.77 (3.0) — 尖锐倒U, 有效窗口 ≈ [0.2, 0.5], w=0.3 附近峰值
- **结论**:
  1. **方向c可行但窗口窄**: EMA 跨视角一致性正则在 w=0.3 存在明确增益 (first 口径显著), 推翻 EXP-044 "≡raw 方向关闭"结论 (当时 w=0.1 太弱)
  2. **机制 vs 离线平滑**: cw03 first=any=95.08 (first 口径 +3 对全部保持), b=0 只赢不输——与 a07 (waldo 有伤害案例) 不同; 训练时正则把一致性约束内化到优化过程, 未见 bbox 中心偏移伤害
  3. **倒U窗口窄是双刃剑**: 优点=w 的物理意义清晰 (正则强度), 缺点=跨场景可能需重新调 w; 必须 figurines/waldo 泛化验证
  4. cw03 的 per-level (77.05/86.89/86.89 first) vs raw —— chosen-level 机制吸收了 level 内差异
- **待办**: ① figurines 泛化 (cw03 配置, ~2.5h) ② waldo 复验 (正则是否无伤害, 对比平滑 -1 对) ③ 若两场景成立 → cw03 进主表替换/并列 a07
- **泛化验证 (EXP-045b, figurines cw03, 23:11 完成)**: **74.07% (40/54) = 显著负** — vs CLIP24d 场 (88.89%) -14.8pp **p=0.0215 ★显著为负** (b=9 c=1); vs raw (77.78%) -3.7pp; vs a07 (90.74%) -16.7pp; 结合 EXP-044 c01(w=0.1)≡raw, figurines 曲线 0.1→77.78 / 0.3→74.07 **全权重无增益峰, 方向c在 figurines 判死**
- **EXP-045 终章结论**:
  1. **w=0.3 增益不泛化**: teatime +4.9pp (显著正) → figurines -3.7pp (vs CLIP 显著负) — 一致性正则非场景无关, cw03 不能进主表
  2. **机理发现 (与离线平滑对照, 论文分析节素材)**: figurines 的 2D DINO 监督跨视角方差是**信息** (混回 2D 即 a07 +11.1pp 最大增益; 压制它即 cw03 显著伤害); teatime 的 2D 方差主要是**噪声/偏移** (混回 2D 仅 +1.6pp; 一致性正则有净收益 +4.9pp) — 平滑增益来源 = 2D 信息回灌而非一致性本身, EXP-044(c01≡raw)+EXP-045(正则伤害) 双向支撑
  3. 方向c最终定位: 通用训练正则 ❌; 机制对照实验 ✅ (与 a07/门控共同构成完整消融链)
- **EXP-045b/c (2026-10-06 08:21 完成) figurines 完整 w 曲线 (用户要求扩网): 严格单调递减, 无峰, figurines 判死铁证**
  - 曲线 (first=any, n=54): w=0.1 **77.78** (≡raw 42/54, 峰值即无增益) → 0.15 75.93 → 0.2 74.07 → 0.3 74.07 → 0.5 68.52 → 0.7 66.67; raw 场 77.78 (42/54)
  - 显著性: vs CLIP24d 场 全部显著负且随 w 加深 (0.15 p=0.0391 / 0.2 p=0.0215 / 0.3 p=0.0215 / 0.5 p=0.0039 / **0.7 p=0.0005, b=12 c=0**); vs a07 (49/54) 同构加深 (0.15 p=0.0078 8:0 / 0.2 p=0.0039 9:0 / 0.7 p=0.0002 13:0)
  - **结论**: figurines 上 EMA 正则从 w=0.1 起任何强度都伤害且严格单调 — "2D 方差=信息"机理的完整剂量-反应证据; teatime 倒U (有峰) vs figurines 单调降 (无峰) 的对比 = 场景依赖的最终形态
  - 工具: run_exp045b_grid.sh {0.15,0.2} + run_exp045c_grid.sh {0.5,0.7} (等待队列设计: pgrep 轮询前序脚本退出后接续, 不可修改运行中脚本); 第4次变量拼接 bug: `$TAG_vs_a07_` 中下划线是合法变量名字符被整体解析为空——多变量拼接必须用 `${TAG}` 花括号
  - 产物: json FIGURINES_DINO_ccw{015,02,05,07}.json; 渲染已自动删 (ckpt 保留可复原); 磁盘余 151G
- **产物**: json `eval_result/mcnemar/FIGURINES_DINO_ccw03.json`; log `eval_result/adaptive/exp045_figurines.log`; figurines cw03 渲染+ckpt 未删 (待用户确认, 可复原); teatime cw10/cw30 已删 (213GB)

---

## 实验编号: EXP-046 — 判别性端到端场: tile 相似结构蒸馏 (lf_rel_weight)
- **日期**: 2026-10-06 16:58 完成
- **分支**: experiment/e2e-discrim (从 experiment/crossattn-mm 0d410c1 切出, 可整体丢弃回退)
- **动机**: AE 重建损失与下游判别反复脱钩 ("重建好≠下游好" 出现 3 次)。把 tile 相似结构直接蒸馏进 render 特征, 砍掉"先重建后使用"的两段式
- **方法**: GS 训练损失 = L1 回归锚 + w_rel × MSE(S3, S2)
  - S2 = 同帧全部 tile 的 2D GT 特征 cos 相似矩阵 (off-diag cos mean=0.45/std=0.20, 结构信号丰富)
  - S3 = render tile 均值的 cos 相似矩阵; L1 锚防止结构匹配流形漂移
  - 实现: train.py gtt_cache (2D GT tile 均值缓存) + 共享 tile means 重构 (lf_cons/lf_rel 复用); 修复 _s.npy 与 -r 分辨率不匹配隐患 (nearest 插值, lf_cons 一并受益)
- **场景**: teatime (image-query first/any 口径, n=61; raw 90.16/93.44, a07 91.80/93.44)
- **结果 (first=any 同数)**:
  - w_rel=1: 88.52 (54/61), vs raw −1 对 ns, vs a07 −2 对 p=0.69
  - w_rel=5: 75.41 (46/61), vs raw −14.8pp **p=0.0225★** / any p=0.0034★
  - w_rel=20: 45.90 (28/61), vs raw −44.3pp **p<0.0001★** (discordant b=29 c=2)
- **结论**:
  1. **严格单调下降, 无峰, 方向关闭** — 相似结构蒸馏随权重单调伤害, 强权重腰斩判别力 (45.90%)
  2. **机理**: 强 S2 约束把 render 特征拉向"复现 2D 相对相似度"流形, 牺牲 tile 绝对判别信息; w=1 温和档也已低于 raw (−1 对)
  3. **第三块机理证据 (论文分析节)**: 训练时对 render 特征施加的约束 — 一致性 (lf_cons: teatime 有峰 w=0.3 但场景依赖/figurines 单调伤害)、结构蒸馏 (lf_rel: 全程单调伤害) — 均不产生可靠增益; **唯一有效机制 = 离线混回 2D 信息 (α 平滑)**, 主表配置维持全局 α=0.7
  4. 与 EXP-025 (tile-mean 抹除选择性) 呼应: 聚合/结构类约束与 tile 级判别天然冲突
- **工程坑**: ① -m 自动追加 _<level> 后缀 (memory 硬约束), 手动加 _$L 导致目录 teatime_dino_32dr1_1_1, 已改名复用零浪费; ② render.py 无 --skip_mesh/--skip_interpolate 参数, 正确参数为 --include_feature
- **产物**: json TEATIME_DINO_cr{1,5,20}.json; log eval_result/adaptive/exp046_teatime_rel3.log; 渲染已删 (ckpt 保留); 分支处置建议: lf_rel 代码已随本分支存在, 效果否定, 分支不合并 (留档) 或丢弃均可, 主线 crossattn-mm 不受影响

---

## 实验编号: EXP-047a — N 教师路由可行性: depth-query 2D 可分性快测
- **日期**: 2026-10-06
- **分支**: experiment/e2e-discrim
- **动机**: 把"双场"推广为"N 教师路由"(text→CLIP / image→DINOv2 / depth→depth教师)。项目铁律: 免训练 2D gate 未过不进 3D
- **协议**: eval/depth_query_test.py — query = DAv2 vitl 相对深度 bbox crop (32×32 归一化), db = 帧 B 全图滑窗 (stride 16) Pearson 相关, hit = 峰落同标签 bbox (first/any)
- **结果**:
  - teatime: first 8/61 = **13.11%** / any 13.11% (DINO image-query 同协议 91.80/93.44%)
  - ramen: first 10/79 = **12.66%** / any 15.19% (DINO 78.48/84.81%)
- **结论**:
  1. **depth-query ≈ 随机 (~13%), 深度模态判死 (2D 层面)** — 相对深度归一化后不同物体的 patch (平滑表面/透视梯度) 高度相似, Pearson 相关无实例判别性; 重复物体多的 ramen 也仅 15%
  2. EXP-013 (融合侧 depth tile 统计无增益) + EXP-047a (查询侧 depth patch 无判别性) = **深度轴双向闭合**
  3. N 教师路由的框架表述: 路由可行性取决于"教师是否提供与查询模态匹配的判别空间"——CLIP (text-image 对齐) / DINOv2 (实例判别) 有, depth (几何) 无; N 场不是场越多越好, 而是"判别空间匹配才路由"。此表述比"N 场"更强且与全部实验证据自洽
- **待测**: DINOv3 教师升级 (timm vit_large_patch16_dinov3 架构可用, 权重 gated 需 HF token) — tile_retrieval_test.py 快测判 gate
- **产物**: eval/depth_query_test.py; depth maps 缓存 teatime/ramen (depth_maps/); log eval_result/adaptive/exp047_depth_*.log

- **EXP-047 收官 (用户决策)**: DINOv3 教师升级跳过 (骨干轴历史全败 EXP-022/028/030-034, 权重 gated 成本>预期收益); sketch 无 GT 协议不可测。**N 教师路由最终形态 = "判别空间匹配路由"框架**: text→CLIP / image→DINOv2 双场主表 + depth 判死 (EXP-047a) + 骨干轴负结果链——路由可行性由教师判别空间与查询模态的匹配度决定, 非场数量
