# EXP-001~027 结论链梳理与论文实验设计

> 日期: 2026-09-28
> 范围: EXP-001~027 全部实验 (分支 experiment/crossattn-mm)
> 数据来源: hyper_parameter.md (实验全记录) / current_situation_LangSplat.md / engineering_experience_LangSplat.md
> 场景: teatime (177 帧, 6 GT 帧), figurines (299 帧, 4 GT 帧); 评估口径: mIoU (multi-prompt mean) / mAcc

---

# 一、结论链总览

## 阶段 0-1: AE 瓶颈发现与融合最优化 (teatime, EXP-001~012)

| 关键节点 | 实验 | 发现 |
|------|------|------|
| 融合起点 | 002-004 | PCA blend (-5.8pp)、cross-attn (-10.8pp) 全败——但这是 **3d AE 瓶颈**造成的假象 |
| 瓶颈解锁 | 005-008 | 3d→6d→12d, fused 差距 -10.8% → -1.8% → -0.5%; **3d 不成比例地惩罚 fused** |
| 最优组合 | 009-010 | InfoNCE 崩 (0.5270, 偏离 CLIP 空间); **mse_only + 12d = 0.6723/0.9153, 首次双超 baseline** |
| 维度上限 | 011-012 | baseline 12d 达峰后退化 (16d 0.6401); fused 16d 饱和; **fused 对维度更鲁棒** |

细节对照表 (teatime, IoU chosen / Localization):

| 方案 | AE dim | IoU | Loc |
|------|--------|------|------|
| baseline (纯 CLIP) | 3 | 0.6431 | 0.8983 |
| baseline | 6 | 0.6285 | 0.8644 |
| baseline | 12 | 0.6660 | 0.8814 |
| baseline | 16 | 0.6401 | 0.8475 |
| cross-attn fused (mse_cos) | 3 | 0.5351 | 0.8814 |
| cross-attn fused (mse_cos) | 6 | 0.6251 | 0.9153 |
| cross-attn fused (mse_cos) | 12 | 0.6609 | 0.8644 |
| cross-attn fused (infonce) | 12 | 0.5270 | 0.8475 |
| **cross-attn fused (mse_only)** | **12** | **0.6723** | **0.9153** |
| cross-attn fused (mse_only) | 16 | 0.6698 | 0.8814 |

贯穿规律: **"AE 重建好 ≠ 下游好"** (AE loss 更低但 IoU 更差的情况出现 3 次: EXP-004 融合网络 / EXP-011 16d / EXP-015 depth aux), 重建指标不能替代下游评测。

## 阶段 2: 第三模态结构性关闭 (EXP-013~016)

- 加入 Depth Anything V2 (8 维 tile 统计): 0.6532 (-1.9pp vs 双模态最优 0.6723)
- 假设1 "AE 维度不足": 16d 0.6456 → 否定
- 假设2 "损失未针对设计": 辅助损失 λ 单调链条 λ=1.0 (0.1646 灾难) → λ=0.1 (0.5588) → λ=0 (0.6532) → 否定
- **结论: 深度保留与 CLIP 兼容性此消彼长, 结构性失败**, 不是超参问题。辅助损失把 fused 拉离 CLIP 空间 (cos_sim 0.98→0.86), 下游崩塌与 InfoNCE 同一失败模式

## 阶段 3: 跨场景验证——增益的分化 (EXP-017~021)

| 指标 | teatime (EXP-010 vs 007) | figurines (EXP-017) | 判定 |
|------|---------|-----------|------|
| fused IoU 增益 | +0.6pp | **-1.9pp (符号翻转)** | 不泛化 |
| fused Loc 增益 | **+3.4pp** | **+1.8pp** | 两场景一致 |

维度扫描全貌 (IoU chosen):

| 场景 | baseline 曲线 | baseline 峰值 | fused 曲线 | fused 峰值 |
|------|--------------|--------------|------------|-----------|
| teatime | 3d 0.6431 / 6d 0.6285 / 8d **0.6705** / 12d 0.6660 / 16d 0.6401 | 8d (倒 U) | 8d 0.6375 / 12d **0.6723** / 16d 0.6698 | 12d |
| figurines | 12d 0.5466 / 16d 0.5622 / 20d 0.5668 / 24d **0.5751** | 24d (未见顶, 单调升) | 12d 0.5274 / 16d 0.5547 / 20d **0.5744** / 24d 0.5447 | 20d (倒 U 更陡) |

- 维度最优值**强场景相关**: teatime 8d / figurines 24d+ (baseline); teatime 12d / figurines 20d (fused)
- 融合领先只存在于 figurines 20d 附近的窄窗口, 更高维 baseline 反超
- 8d 符号翻转 (teatime): baseline 8d 反超 fused——维度选择取决于特征类型, 纯 CLIP 可更低维, DINOv2 融合需 ≥12d

## 阶段 4-5: 骨干/架构关闭 + 查询协议唯一正收益 (EXP-022~024)

- **EXP-022 CLIP ViT-L/14 (768d) 骨干升级**: 两场景 -7pp → 瓶颈不在骨干容量, 在 tile 聚合协议与查询层; 768d 高压缩率 (32:1) 判别性更差, 保持 B/16
- **EXP-023 双流拼接 (CLIP 8d + DINO 8d, score 级 late fusion)**: λ 单调劣化 (0.6705 → 0.5536@λ0.3 → 0.6107@λ0.5) → DINO 投影流 = 纯噪声; 对齐 cos_sim 0.9256 只证整体相似不证判别力
- **三路线大汇聚 (InfoNCE / 深度三模态 / 双流)**: LangSplat 的 text-query 协议下, **CLIP 空间是唯一有效语义空间, 一切注入均稀释 softmax(10·sims) 尖峰分布**
- **EXP-024 查询协议 (唯一免训练正收益)**: 7 模板 ensemble **+2.3pp (teatime) / +1.5pp (figurines)**; 温度 10 双场景最优且鲁棒 (±0.4pp)
  - **新 SOTA: teatime 8d+ensemble 0.6932/0.8983, figurines 24d+ensemble 0.5905/0.8214**
  - figurines 0.5905 已逼近当时认知的 2D 上限 0.593
  - 评估口径统一: 后续全部报 mIoU (multi-prompt mean) / mAcc

## 阶段 6: 密集化三连关 (EXP-025~027, 免训练 2D 上限)

| 尝试 | 失败机理 | 数字 (teatime 2D tile mIoU) |
|------|---------|------|
| EXP-025 tile-mean dense (MaskCLIP v-proj 对 tile 内 ~196 patch 取均值) | 判别方差坍缩 4× (pos_sim std 0.010 vs global 0.037) | 0.0321 / +模板 0.0242 (vs global 0.4399/0.5394, -40pp) |
| EXP-026 per-patch dense (单 resize 14×14 / 滑窗 per-pixel, MaskCLIP 正确形态) | **空间错位**——方差保留 (std 0.0266) 但激活与 GT 无关 (stuffed bear IoU≈0, tile 级 0.97) | 0.0096~0.0191 |
| EXP-027 背景抑制填充 (gray/black/blur) | 原管线已是最优 (mask 外置黑+pad); blur 无模板 +4.4pp 但与 ensemble 冲突 | blur 0.4837/+模板 0.4705 < 基线+模板 0.5394 |

两个元结论:

1. **CLIP 模型本身不产 dense 判别特征** (对比训练只在 global image-text 层对齐, patch 级 text 对齐是未训练副作用) → 修正 EXP-022 的判断: tile global embedding 不是瓶颈, 是 **CLIP 约束下的最优解**
2. **3D 渲染特征 >> 2D tile map** (0.6932 vs 0.5394, +23pp)——AE 压缩+多视角融合+per-pixel 渲染远强于分段常数 tile 图; "2D 上限"实为下限参考

EXP-027 附加发现: **图像端编码改进与 text 端查询协议改进不可叠加** (此消彼长), 现行"黑背景+pad × 7模板 ensemble"已是该协议下的局部最优。

## 一句话结论链

> 特征注入 (blend / cross-attn / InfoNCE / 深度三模态 / 双流 / L14 骨干 / dense×3 粒度 / 背景填充) 九路全败, 败因同源: **CLIP 语义空间的封闭性**; 有效增益只有两处——**融合的定位价值** (Loc 跨场景一致 +1.8~3.4pp) 和**查询协议** (ensemble 免训练 +1.5~2.3pp); 而 3D 多视角融合本身已远超 2D 特征质量上限。

---

# 二、论文实验设计讨论

## 手里真正可发表的资产

| 资产 | 强度 | 缺口 |
|------|------|------|
| ① 融合提升**定位** (IoU 不稳但 Loc 一致) | 中强 (两场景一致) | 需 waldo_kitchen/ramen 第三、第四场景确认 |
| ② 查询协议 ensemble 免训练增益 | 强 (两场景一致、零成本、可叠加) | 同上; 单独撑不起一篇论文 |
| ③ 维度-场景相关性 (fused 鲁棒 vs baseline 敏感) | 中 | 需在另外两场景复现趋势 |
| ④ 系统性负结果地图 (9 条路线 + 机理诊断) | **独特** (社区几乎没人做) | 需要更规范的实验设计包装 |

## 三个候选叙事

**A. 实证分析论文** ("An Empirical Study of Language-Embedded 3DGS: Where Do the Gains Come From?")
- 故事: 解构 LangSplat 管线的三段 (2D 教师编码 → AE/3D 融合 → 查询决策), 用 27 个实验回答"增益从哪来、为什么别的路都不通"
- 贡献点: 查询协议免费午餐; 3D 多视角融合 > 2D 特征质量 (+23pp 反直觉发现, 很有卖点); CLIP 空间封闭性的机理论证 (方差坍缩 vs 空间错位是两个不同失败模式, 诊断工具可复用); 负结果系统化
- 门槛: 需要 4 场景全覆盖 + 阈值敏感性分析, 工作量主要在补场景

**B. 定位导向论文** ("DINOv2 Fusion Improves Open-Vocabulary 3D Localization")
- 故事: 分割 IoU 不是融合的正确度量, 定位 (Loc) 才是——DINOv2 的互补价值在"物体在哪"不在"边界多准"
- 优点: 单一 positive claim 清晰; 缺点: 依赖 Loc 增益在另外两场景复现, 且 1.8~3.4pp 幅度偏弱, 审稿风险高

**C. 组合叙事 (推荐)**: 以 A 为骨架, 把 B 作为正面结果之一, ②③④ 作为支撑章节。数据先决: **全场景主表决定故事走向**。

## 论文实验矩阵 (补齐后即成文骨架)

| 表/图 | 内容 | 现状 |
|-------|------|------|
| Table 1 主表 | 4 场景 × {baseline 最优维, fused 最优维} × {±ensemble}, 报 mIoU/mAcc | 缺 waldo_kitchen/ramen 全部 |
| Table 2 维度扫描 | 每场景 4~5 个维度曲线 (场景相关性证据) | teatime/figurines 已有 |
| Table 3 融合消融 | mse_cos / infonce / mse_only; ±residual; 深度 λ 扫描 | teatime 已有, 需 1 个场景复现 |
| Table 4 查询协议 | 温度扫描 + 模板数消融 | 部分已有 (模板数消融未做) |
| Fig 上限分析 | 2D tile map vs 3D 渲染 vs GT; dense 失败机理图 (方差坍缩/空间错位) | 数据已有 (EXP-025/026/027) |
| Fig 结论链 | 设计空间地图 (本文第一节) | 可直接画 |

## 必须提前处理的协议弱点 (审稿人会打)

1. **GT 帧太少**: teatime 仅 6 帧 × ~11 类; figurines 4 帧。建议评估全部 LERF-OVS GT 或补充自标注
2. **mask_thresh 按场景手调** (teatime 0.40 / figurines 0.45 / ramen 0.55): 主表必须加阈值敏感性曲线, 或统一自适应阈值
3. **baseline 复现声明**: EXP-001 (AE 3d, 0.6431/0.8983) 与 LangSplat 原文数字的对应关系要写清楚复现口径
4. **ramen 已知坑**: 训练不用 --eval flag (train/test split 导致渲染帧不足, eval IndexError); mask_thresh=0.55

## 下一步建议 (数据决定论)

启动**全场景主表**: waldo_kitchen、ramen 两场景, 各跑 {baseline 8d/12d, fused 12d} × eval (±ensemble), 约 8~10 条训练链。这批数据出来后:

- 若 Loc 增益在 3~4 场景一致 → 叙事 B 成立, 论文有硬正结果
- 若 ensemble 增益全场景一致 → ② 升级为可声明结论
- 无论结果如何 → A/C 的负结果地图都成立

---

## 附: 工程资产清单 (实验产出的可复用工具)

| 工具 | 用途 |
|------|------|
| eval/dense_upper_test.py | 2D tile 级免训练 eval (协议严格对齐 activate_stream, 含积分图版 smooth 100× 加速) |
| eval/dense_patch_upper_test.py | per-patch dense 教师快测 (单 resize / 滑窗 dense map) |
| eval/masked_pool_test.py | tile 编码方案快测 (_s.npy 反推 tile mask, 免重跑 SAM) |
| eval/evaluate_iou_loc.py | mIoU/mAcc 标准输出 + --relev_temp/--prompt_ensemble CLI |
| preprocess.py encode_image_dense | MaskCLIP 式 dense 特征提取 (--dense_tiles/--no_ln_post/--save_subdir) |
| mm_langsplat/train_align.py 等 | 融合训练/双流 eval 全套参数化流程 (EXP-022/023 遗产) |

判别性快测三件套判据 (§8.19): ① pos_sim std (方差) ② 大物体 rel_in vs rel_out (空间对齐) ③ 协议内 IoU——三者全过才值得跑全链条。
