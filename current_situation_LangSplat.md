# LangSplat 多模态融合实验 — 当前进展总结

> 分支: experiment/crossattn-mm
> 场景: LERF teatime (177 张图)
> 基线: LangSplat (CLIP ViT-B/16, laion2b_s34b_b88k)
> 评估指标: IoU chosen, Localization accuracy
> 最后更新: 2026-09-19

---

## 一、研究目标

探索在 LangSplat (开放词汇 3D 场景理解) 框架中，融合 CLIP (语义) 与 DINOv2 (几何/纹理) 多模态特征能否提升 3D 开放词汇分割性能 (IoU + Localization)，并找出最优的融合架构、损失函数和 AE 瓶颈维度组合。

---

## 二、实验演进路线

### 阶段一：简单融合方案 (EXP-001~003) — 均失败
| 实验 | 方案 | IoU | vs baseline | 结论 |
|------|------|-----|-------------|------|
| EXP-001 | baseline (纯CLIP) 3d | 0.6431 | - | 基线 |
| EXP-002 | PCA blend01 (90%C+10%D) | 0.5854 | -5.8% | 简单混合稀释 CLIP 语义 |
| EXP-003 | PCA blend02 (80%C+20%D) | 0.5844 | -5.9% | DINOv2 比例越高越差 |

**阶段结论**: 简单线性混合 (PCA 降维 + 加权) 稀释 CLIP 语义判别力，不可行。

### 阶段二：Cross-Attention 融合 (EXP-004) — 3d 下失败
| 实验 | 方案 | IoU | vs baseline | 结论 |
|------|------|-----|-------------|------|
| EXP-004 | tile级 cross-attn fused 3d | 0.5351 | -10.8% | 比 blend 更差 |

**关键诊断**: 融合网络 cos_sim=0.98 → 主要在重建 CLIP，丢失语义判别力。AE 重建 loss 低 (0.189) 不代表下游好。

### 阶段三：AE 瓶颈验证 (EXP-005~008) — 找到突破口
| 实验 | 方案 | AE dim | IoU | 关键结论 |
|------|------|--------|-----|---------|
| EXP-005 | baseline 6d | 6 | 0.6285 | 纯CLIP 6d 反而下降 |
| EXP-006 | fused 6d | 6 | 0.6251 | fused +9.0%，Loc 超基线 |
| EXP-007 | baseline 12d | 12 | 0.6660 | CLIP 更优维度 |
| EXP-008 | fused(mse_cos) 12d | 12 | 0.6609 | 差距缩至 -0.5% |

**关键发现**: **3d AE 瓶颈是不成比例损害 fused 特征的根本原因**。fused IoU 随维度单调上升 (0.5351→0.6251→0.6609)，差距从 -10.8% 缩至 -0.5%。推翻"DINOv2 无增益"结论。

### 阶段四：损失函数改进 (EXP-009~010) — 突破 baseline
| 实验 | 损失函数 | 融合 cos_sim | IoU | Loc | vs baseline 12d |
|------|---------|-------------|-----|-----|-----------------|
| EXP-009 | InfoNCE | 0.5810 | 0.5270 | 0.8475 | -13.9% |
| EXP-010 | **MSE-only** | 0.9796 | **0.6723** | **0.9153** | **+0.6% / +3.4%** |

**关键发现**:
- **MSE-only 首次超越 baseline** (IoU 0.6723 > 0.6660)
- 损失排序: mse_only > mse_cos > infonce
- cos_sim 项对 unit-normalized 输出与 MSE 近似等价，属冗余约束
- InfoNCE 对比学习使 fused 过度偏离 CLIP 空间，AE 重建困难，下游 IoU 大幅下降

### 阶段五：更高 AE 维度 (EXP-011~012) — 12d 确认为最优
| 实验 | 方案 | AE dim | IoU | Loc | 关键结论 |
|------|------|--------|-----|-----|---------|
| EXP-011 | baseline 16d | 16 | 0.6401 | 0.8475 | baseline 超过 12d 后退化 |
| EXP-012 | fused(mse_only) 16d | 16 | 0.6698 | 0.8814 | fused 饱和略降，但仍超 baseline +3.0% |

**关键发现**: 16d 确认 12d 为最优 AE 维度（fused 饱和，baseline 退化）。但 fused 对维度更鲁棒——16d 下优势扩大至 +3.0pp IoU（12d 时仅 +0.6pp）。fused IoU 单调上升至 12d 后饱和；baseline 在 12d 达峰后退化。

---

## 三、当前最优结果

| 方案 | AE dim | 损失函数 | IoU | Loc |
|------|--------|---------|-----|-----|
| **cross-attn fused (MSE-only)** | **12** | **mse_only** | **0.6723** | **0.9153** |

vs 基线 (baseline 3d): IoU +2.9%, Loc +1.7%
vs baseline 12d: IoU +0.6%, Loc +3.4%

---

## 四、完整结果汇总表

| 方案 | AE dim | IoU chosen | Localization | vs baseline 3d IoU |
|------|--------|-----------|--------------|---------------------|
| baseline (CLIP) | 3 | 0.6431 | 0.8983 | - |
| baseline (CLIP) | 6 | 0.6285 | 0.8644 | -1.5% |
| baseline (CLIP) | 12 | 0.6660 | 0.8814 | +2.3% |
| baseline (CLIP) | 16 | 0.6401 | 0.8475 | -0.3% |
| blend01 (90%C+10%D) | 3 | 0.5854 | 0.8814 | -5.8% |
| blend02 (80%C+20%D) | 3 | 0.5844 | 0.8644 | -5.9% |
| cross-attn fused (mse_cos) | 3 | 0.5351 | 0.8814 | -10.8% |
| cross-attn fused (mse_cos) | 6 | 0.6251 | 0.9153 | -1.8% |
| cross-attn fused (mse_cos) | 12 | 0.6609 | 0.8644 | -0.5% |
| cross-attn fused (infonce) | 12 | 0.5270 | 0.8475 | -13.9% |
| **cross-attn fused (mse_only)** | **12** | **0.6723** | **0.9153** | **+2.9%** |
| cross-attn fused (mse_only) | 16 | 0.6698 | 0.8814 | +2.7% |

---

## 五、核心结论

1. **DINOv2 融合确实有增益**，但需三个条件同时满足：
   - 足够的 AE 维度 (12d，非 3d/6d；16d 已饱和)
   - 合适的损失函数 (MSE-only，非 mse_cos/infonce)
   - cross-attention 架构 (非简单 PCA blend)

2. **AE 瓶颈是关键变量**: 3d 瓶颈不成比例损害 fused 特征。fused 受益于高维度，baseline 反而在 6d 下降。16d 实验确认 12d 是最优维度：fused 饱和 (0.6698 vs 0.6723)，baseline 退化 (0.6401 vs 0.6660)。

3. **损失函数设计重要**: cos_sim 项冗余 (MSE≈2*(1-cos_sim))，去掉后网络更自由学习。InfoNCE 过度偏离 CLIP 语义空间。

4. **AE 重建好 ≠ 下游好**: 融合网络 cos_sim 高 (0.98) 说明在复制 CLIP，不代表学到有用融合；AE loss 低也不代表下游 IoU 好（16d 再次验证: baseline AE loss 0.1377<0.1469 但 IoU 更差）。

5. **fused 对 AE 维度更鲁棒**: 维度从 12d→16d，baseline 退化 -2.6pp，fused 仅 -0.25pp，fused 相对优势扩大至 +3.0pp IoU / +3.4pp Loc。

---

## 六、已完成的工作

- ✅ 多模态特征提取 pipeline (CLIP + DINOv2 + SAM tile)
- ✅ tile 级 cross-attention 融合网络 (3 种损失函数)
- ✅ AE 维度支持 (3d/6d/12d/16d，rasterizer + 全链路修改)
- ✅ 12 组实验 (EXP-001~012) 完整跑通并记录
- ✅ 工程经验文档 (engineering_experience_LangSplat.md)
- ✅ 代码与结果已 git 提交 (commit ff0400d；EXP-011/012 待提交)

---

## 七、可能的后续方向

1. **跨场景验证**: 当前仅在 teatime 验证。需在 figurines/waldo_kitchen/ramen 上验证 MSE-only+12d 是否稳定超越 baseline。

2. **更高 AE 维度**: ~~16d 已验证饱和~~ (EXP-011/012: fused 0.6698 < 12d 0.6723，baseline 退化)。24d 大概率继续退化，不建议尝试；如需验证可在其他场景做。

3. **融合架构改进**: 当前 tile 级 cross-attn。可尝试更细粒度 (pixel 级，需解决 OOM) 或更粗粒度 (scene 级 global context)。

4. **DINOv2 变体**: 当前用 ViT-B/14 (768d)。可尝试 ViT-L/14 (1024d) 或 ViT-S/14 看是否有差异。

5. **损失函数组合**: MSE-only + 轻度正则 (如 KL 散度约束分布) 可能在保持 CLIP 对齐的同时引入判别力。

6. **多模态扩展**: 除 DINOv2 外，可融合深度/法线/纹理等几何特征 (extractors 已实现但未用于融合)。
