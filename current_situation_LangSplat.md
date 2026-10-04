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
- ✅ 16 组实验 (EXP-001~016) 完整跑通并记录
- ✅ 工程经验文档 (engineering_experience_LangSplat.md)
- ✅ 代码与结果已 git 提交 (commit ff0400d；EXP-011/012 已提交 56dcab2；EXP-013 三模态融合已完成)

---

## 七、可能的后续方向

1. **跨场景验证**: 当前仅在 teatime 验证。需在 figurines/waldo_kitchen/ramen 上验证 MSE-only+12d 是否稳定超越 baseline。

2. **更高 AE 维度**: ~~16d 已验证饱和~~ (EXP-011/012: fused 0.6698 < 12d 0.6723，baseline 退化)。24d 大概率继续退化，不建议尝试；如需验证可在其他场景做。

3. **融合架构改进**: 当前 tile 级 cross-attn。可尝试更细粒度 (pixel 级，需解决 OOM) 或更粗粒度 (scene 级 global context)。

4. **DINOv2 变体**: 当前用 ViT-B/14 (768d)。可尝试 ViT-L/14 (1024d) 或 ViT-S/14 看是否有差异。

5. **损失函数组合**: MSE-only + 轻度正则 (如 KL 散度约束分布) 可能在保持 CLIP 对齐的同时引入判别力。

6. **多模态扩展**: ~~可融合深度/法线/纹理等几何特征~~ (EXP-013~016 已闭环验证: 深度第三模态 IoU 0.6532 < 双模态 0.6723；AE 提到 16d 更差 (0.6456)；深度辅助损失 λ=1.0/0.1 均劣化 (0.1646/0.5588)。两个假设 (AE维度不足/损失未针对深度设计) 均被实验否定，负结果为结构性: fused 必须保持 CLIP 兼容，深度几何与 CLIP 语义判别根本冲突。**深度方向关闭，不建议再投入**。)


## 9. EXP-017: figurines 跨场景验证 (2026-09-21)
### 9.1 结果
- baseline 12d: IoU=0.5466, Loc=0.7321；fused(mse_only) 12d: IoU=0.5274, Loc=0.7500
- fused IoU 增益跨场景翻转 (teatime +0.6pp → figurines -1.9pp)；Loc 增益两场景一致 (+3.4pp/+1.8pp)
- **结论: DINOv2 融合对定位 (粗粒度) 有跨场景稳定价值，对分割 IoU (细粒度) 无**

### 9.2 诊断插曲 (详见 engineering_experience §8.12)
- 首次 figurines eval IoU=0.0194 崩溃，根因为 dim12 编码早于 AE 收敛 (竞态)，非管线/视角/CLIP 问题
- 用最终 AE ckpt 重编码 → 重训 baseline 3GS → 重渲染 → 重评，数字恢复正常
- 三源对照法 (2D原始 / AE往返 / 3D渲染 = 0.593/0.063/0.025) 是定位此类崩溃的高效手段

### 9.3 下一步方向建议
1. DINOv2 融合若继续推进，价值主张应转向定位任务 (Localigation)，或探索仅在 encoder 侧用 DINOv2 而不进入 CLIP 空间
2. 深度模态方向维持 EXP-013~016 结论：结构性关闭 (除非改架构位置，如几何正则)
3. figurines 场景 2D CLIP 单帧上限 0.593，说明瓶颈部分在 CLIP 特征本身对密集小物体的分辨力，可探索更高分辨率 tile 或 SAM mask 级特征聚合

## 10. EXP-018: AE 维度 8d 验证 (2026-09-21)
### 10.1 结果 (teatime)
- baseline 8d: IoU=**0.6705**, Loc=0.8814 → **超过 12d (0.6660)，baseline 新最优**
- fused(mse_only) 8d: IoU=0.6375, Loc=0.8644 → 明显低于 12d (0.6723)，-3.5pp
- 8d 时符号翻转：baseline 反超 fused (-3.3pp)
### 10.2 结论
- 纯 CLIP 最优维度是 8d（非 12d）：8d > 12d > 3d > 16d > 6d
- DINOv2 融合需要 ≥12d：融合特征信息量更大，低 AE 维度优先损失融合增益
- fused 对维度下限敏感 (12d→8d -3.5pp)、对上限鲁棒 (12d→16d 仅 -0.25pp)
- 后续如继续融合方向，AE 维度锁定 12d；纯 CLIP baseline 建议改用 8d

## 11. EXP-019: figurines 跨场景 16d 验证 (2026-09-21)
### 11.1 结果 (mask_thresh=0.45)
- baseline 16d: IoU=0.5622, Loc=0.7857（vs 12d: IoU +1.6pp, Loc +5.4pp）
- fused(mse_only) 16d: IoU=0.5547, Loc=0.7500（vs 12d: IoU +2.7pp, Loc 持平）
### 11.2 结论
- **维度最优值场景相关**: teatime 16d 双双退化、8d 最优；figurines 16d 双双提升仍在涨——AE 维度需按场景调优，无全局最优
- 融合增益跨场景依然不稳定（teatime 16d +3.0pp / figurines 16d -0.75pp），融合价值主张仍限于定位（且 figurines 16d Loc 也翻转了，定位优势亦不稳固）
- 16d baseline figurines Loc 0.7857 创该场景新高

## 12. EXP-020: figurines 20d 验证 (2026-09-22)
### 12.1 结果 (mask_thresh=0.45)
- baseline 20d: IoU=0.5668, Loc=0.7679（vs 16d: IoU +0.5pp 趋缓, Loc -1.8pp 回落）
- fused(mse_only) 20d: IoU=0.5744, Loc=0.7857（vs 16d: IoU +2.0pp, Loc +3.6pp）
### 12.2 结论
- **fused 20d 首次在 figurines IoU+Loc 双超 baseline**，0.5744 创该场景全实验新高
- fused figurines 维度曲线单调上升 (12d 0.5274 → 16d 0.5547 → 20d 0.5744)，与 teatime 完全相反——维度需求场景相关，figurines 密集小物体需更高维度
- baseline 20d 接近饱和（IoU 趋缓、Loc 回落），fused 在高维仍有明显增益
- DINOv2 融合的价值在高维 + 复杂场景下成立；可探索 24d 确认 fused 峰值

## 13. EXP-021: figurines 24d 验证 (2026-09-22)
### 13.1 结果 (mask_thresh=0.45)
- baseline 24d: IoU=0.5751, Loc=0.8214（IoU 仍在涨, Loc 创全实验新高）
- fused(mse_only) 24d: IoU=0.5447, Loc=0.7500（vs 20d: IoU -3.0pp 回落）
### 13.2 结论
- fused figurines 倒 U 型曲线确认，峰值 20d (0.5744)；fused 最优维度: teatime=12d, figurines=20d
- baseline figurines 曲线单调上升未见顶 (12d 0.5466 → 24d 0.5751)，与 teatime (8d 见峰) 相反
- 融合领先仅存在于 20d 附近窗口，baseline 需更高维度但最终可反超
- 维度最优值强场景相关；fused 对维度更敏感（倒 U 更陡）

- 2026-09-28 EXP-024: 查询协议改进, 7模板 CLIP ensemble 免训练 +2.3pp/+1.5pp, 新 SOTA teatime 0.6932/0.8983, figurines 0.5905/0.8214; 温度10确认最优且鲁棒; 评估口径统一 mIoU/mAcc

## 14. EXP-025: 突破点① dense patch token 2D 上限验证 (2026-09-28)
### 14.1 做法
- MaskCLIP 式 dense 特征 (末层 v-projection, 跳过 q@k) 对每个 SAM tile 内 patch 取均值, 替换 tile 的 CLIP global embedding; teatime 177 帧提取至 language_features_dense/
- eval/dense_upper_test.py 复刻 activate_stream 全协议 (滤波/阈值/majority smooth/chosen level), 6 GT 帧 × 11 prompt 免训练对比
### 14.2 结果 (mIoU chosen-level)
- global: 0.4399 (无模板) / 0.5394 (+7模板)
- dense: 0.0321 / 0.0242 —— **灾难性失败 (-40.8pp / -51.5pp)**
### 14.3 结论
- tile 粒度 mean-pool 抹掉 MaskCLIP 的 patch 级选择性 (pos_sim std 0.010 vs 0.037), 判别方差坍缩; 1 tile=1 向量的聚合粒度未变, 只是把特征换弱
- **重要反直觉发现: 3D 管线 (0.6932) 大幅超越 2D tile map (0.5394)**——AE+多视角融合+per-pixel 渲染比分段常数 tile 图强得多, "2D tile 上限" 框架不成立
- 突破点① 关闭; dense 若要做必须整条替换 tile 协议 (per-pixel 监督/蒸馏), 工程量大暂缓
- 下一步: 查询协议 (EXP-024) 已兑现 +2pp; 突破点③ (更强教师蒸馏) 或全场景主表 (waldo_kitchen/ramen)

- 2026-09-28 EXP-025: dense tile-mean 2D 验证 -40pp 失败关闭; 3D 渲染特征 >> 2D tile map (+23pp), "2D上限"框架失效

## 15. EXP-026: 突破点③ gate — CLIP dense per-patch 教师 (2026-09-28)
### 15.1 结果 (免训练 2D 上限, mIoU)
- single-resize 14×14 patch map: 0.0096 / +模板 0.0056
- 滑窗 per-pixel dense map (224 crop stride 112): 0.0191 / +模板 0.0069
- tile 基线: 0.4399 / 0.5394 —— per-patch 差 40+pp
### 15.2 机理 (与 EXP-025 不同)
- patch 级判别方差不坍缩 (pos_sim std 0.0266 ≈ tile 0.0372), 但空间激活与 GT 完全错位 (stuffed bear IoU≈0, tile 级 0.97)
- 根因: CLIP 对比训练只在 global 层对齐 image-text, patch 级 text 对齐是未训练副作用, 无空间 grounding
### 15.3 结论
- 突破点③轻量形态 (CLIP 系教师 + 换聚合粒度) 全线关闭: EXP-025 (tile-mean, 方差坍缩) + EXP-026 (per-patch, 空间错位) 双实验覆盖全部粒度
- 核心新认知: **问题不在聚合协议, 在 CLIP 模型本身不产 dense 判别特征**——修正 EXP-022"瓶颈在 tile 聚合协议"的判断: tile 聚合协议是"在 CLIP 约束下的最优解"而非瓶颈
- 密集对齐教师 (FC-CLIP/SAN) 重型路线风险大 (协议适配 + EXP-022 前车之鉴), 暂缓
- 下一步: 全场景主表 (waldo_kitchen/ramen, SOTA 协议) 或 masked CLIP pooling (tile crop 背景污染修复, 最后一个轻量教师改进点)

- 2026-09-28 EXP-026: per-patch dense 教师 2D 上限 0.01~0.02 全灭, 空间错位 (非方差坍缩); CLIP 系 dense 教师全线关闭, 突破点③轻量形态关闭

## 16. EXP-027: masked CLIP pooling 快测 (2026-09-28)
### 16.1 发现
- **原管线已是 masked CLIP** (get_seg_img mask 外置黑 + pad_img + resize 224); none 对照掉 19pp 证实必要性
- blur (背景保留模糊) 无模板 +4.4pp (0.4837), black +2.3pp (0.4626)
- 但 +模板后全部低于基线 (blur 0.4705 / black 0.5084 vs 基线 0.5394)——ensemble 收益依赖黑背景特征
### 16.2 结论
- 图像端填充方案与 text 端模板 ensemble **不可叠加**, 现行组合已是局部最优; 2D tile 上限 0.5394 未被打破
- 教师信号轻量改进空间耗尽; gate 未过, 不进 3D
- 顺手修复 eval/openclip_encoder.py encode_image 死代码 mask 转发炸弹

- 2026-09-28 EXP-027: tile 背景抑制方案对比, 原管线黑背景+pad 已最优, blur 无模板+4.4pp 但与 ensemble 冲突, 方向关闭

## §17 方向① 任务路由双特征场 (EXP-035a, 2026-09-29)
- 动机: 34 个实验的负结果链(CLAIP 空间封闭/2D-3D 脱钩/空间错位) + 被低估的正信号(fused Loc 两场景一致增益) → 按查询模态解耦语义场: text→CLIP 场(分割), image→DINO 场(定位)
- 概念验证(teatime): 2D tile 检索 DINO +10.3pp; 3D 场 DINO 32d 90.2% vs CLIP 8d 82.0% (+8.2pp, 61 对象对)
- 与 EXP-023 双流失败的区别: 双流在 text-query 下混合 score; 任务路由不混合, 各任务各走各场, DINO 特征不被挤进 CLIP 空间
- 新工具: eval/tile_retrieval_test.py (2D 快测), eval/eval_image_query.py (3D image-query 检索评测)
- 下一步: figurines 跨场景验证 DINO 场; 64d 动态共享内存改造; 与 LangSplat text-query 分割整合成完整任务路由系统
