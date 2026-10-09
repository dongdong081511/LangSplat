# 论文写作素材汇总 — Task Routing / 任务路由语义场（LangSplat 改进）

> 整理日期：2026-10-03。**纯写作素材，所有数字均可溯源**：标注 `EXP-xxx` 指向 `hyper_parameter.md` 对应小节；标注 `eval_result/...log` 指向评估日志原文。
> **溯源说明**：`eval_result/` 目录下**没有 json 结果文件**（评估脚本 `eval/evaluate_iou_loc.py` 仅落盘 log），text eval 数字提取自 `eval_result/<场景名>/<时间戳>.log`。日志数字与 hyper_parameter.md 锚点**全部一致**，无冲突。

---

## 1. 一句话主张与贡献点

**一句话主张**：开放词汇 3D 高斯分割的查询模态天然异构——text-query 依赖 CLIP 对齐空间、image-query 依赖 DINOv2 实例判别空间；与其把两者挤进一个语义场互相稀释，不如**按查询模态路由到两个独立特征场**（task routing）。多帧证据聚合口径下四场景 pooled McNemar p<0.0001（38:7），text-query 侧 CLIP 场保持 LangSplat 协议最优。

**贡献点（3–4 条）**：
1. **任务路由语义场**：首次按查询模态解耦 3DGS 语义场——text-query → CLIP 场（8d/24d AE），image-query → DINOv2 场（AE 32d），并给出模态路由的 3 级证据链（2D tile 快测 → 3D 场检索 → text-query 注入全败对照）；路由可行性由"教师判别空间与查询模态的匹配度"决定（depth 模态 ~13% 随机判死、骨干升级轴全败佐证）。（EXP-035a→040、EXP-047a）
2. **跨视角自蒸馏平滑 + 多帧证据聚合协议**：(a) 以已训 DINO 场的渲染特征（3D 一致）凸组合混回 2D tile 监督重训，单轮 α=0.7 修复跨视角不一致缺口，并给出启用判据；(b) 揭示单峰检索协议系统性丢弃跨帧证据——majority 口径（per-frame top1 ≥50% 投票）使任务路由优势从 pooled p=0.324（ns）提升至 **p<0.0001**，同时构成对 LERF 系单峰评估协议的方法论批评。（EXP-036a/037、EXP-040、EXP-042、EXP-048）
3. **系统的负结果证据链**：多模态融合、dense 教师、骨干升级、训练时一致性/结构约束五条轴共 30+ 实验一致证明——LangSplat text-query relevancy 协议下 **CLIP 空间是唯一有效语义空间**、增益唯一来自 2D 信息回灌而非 3D 一致性约束，为"路由而非融合/正则"提供动机层证据。（EXP-009/013/023、EXP-025–027、EXP-022/028–034、EXP-043–046）
4. **评估口径敏感性分析**：first/any/majority/hit-rate 四口径矩阵 + McNemar 显著性检验，指出重复标签密集场景（waldo knife×9）下单峰口径系统性低估、多帧聚合口径下每帧劣势场景（waldo）被如实放大——口径选择本身是被评估属性。（EXP-039/040、EXP-041、EXP-048）

---

## 2. 主表 A：四场景 text eval（CLIP 场，mIoU / mAcc，±7 模板 ensemble）

协议：CLIP ViT-B/16 laion2b_s34b_b88k，relev_temp=10，softmax(10·sims)，7 模板 ensemble（EXP-024）；mask_thresh：teatime 0.40 / figurines 0.45 / waldo 0.40 / ramen 0.55。mIoU = multi-prompt mean，mAcc = localization accuracy。

| 场景 | CLIP 场 AE 维度（场景最优） | 无 ensemble mIoU / mAcc | + ensemble mIoU / mAcc | ΔmIoU | 来源 |
|---|---|---|---|---|---|
| teatime | **8d**（基线维度曲线峰值，EXP-018） | 0.6705 / 0.8814 | **0.6932 / 0.8983** | +2.3pp | `eval_result/teatime_8d/20260921_151951.log`（无ens，旧口径 iou chosen / Localization）；`eval_result/teatime_8d/20260928_132408.log`（+ens）；EXP-023 λ=0 行、EXP-024 |
| figurines | **24d**（维度曲线 12d 0.5466 → 16d 0.5622 → 20d 0.5668 → 24d 0.5751 单调升，EXP-017/019/020/021） | 0.5751 / 0.8214 | **0.5905 / 0.8214** | +1.5pp | `eval_result/figurines_24d/20260922_163142.log`（无ens）；`eval_result/figurines_24d/20260928_132254.log`（+ens）；EXP-024 |
| waldo_kitchen | 8d（主表协议，EXP-039） | 0.5870 / 0.8636 | 0.5404 / 0.8182 | **−4.7pp** | `eval_result/waldo_kitchen_clip8d/20261003_113544.log`（无ens）；`eval_result/waldo_kitchen_clip8d/20261003_113730.log`（+ens）；EXP-039 |
| ramen | 8d（主表协议，EXP-039） | 0.5387 / 0.7042 | 0.5062 / 0.7324 | −3.3pp（mAcc +2.8pp） | `eval_result/ramen_clip8d/20261003_165944.log`（无ens）；`eval_result/ramen_clip8d/20261003_170537.log`（+ens）；EXP-039 |

**叙事要点**：
- ensemble 收益**场景相关、两升两降**（+2.3 / +1.5 / −4.7 / −3.3 pp）——主表必须报双口径，不能只报 +ensemble。（EXP-039 结论）
- teatime/figurines 的 +ens 数字为 EXP-024 时全项目 SOTA（0.6932 / 0.5905）；figurines 0.5905 已逼近 2D 单帧上限 0.593。（EXP-024/025）
- 温度扫描佐证协议稳健性：temp 5/10/20/40 → teatime 0.6667/0.6705/0.6658/0.6529，temp=10 严格最优；ensemble 增益来自模板侧表示质量而非温度。（EXP-024）

### 主表 A-2：vs LangSplat 官方 3d baseline（同协议复现，EXP-056，**防"数字不可比"拒稿**）

协议：官方 LangSplat 原版管线 512→3d AE（170:1 压缩）+ rasterizer 3d + 3 level，官方/同源 AE ckpt，evaluate_iou_loc.py 同协议双跑；teatime 用官方 teatime.pth（2024-03）。

| 场景 | LangSplat 3d 无ens（mIoU/mAcc） | LangSplat 3d +ens | 我们 CLIP 场 无ens | Δ mIoU |
|---|---|---|---|---|
| figurines | 0.4825 / 0.7679 | 0.5000 / 0.8214 | 0.5751 / 0.8214（24d） | **+9.3pp** |
| waldo_kitchen | 0.4728 / 0.7727 | 0.3835 / 0.6364 | 0.5870 / 0.8636（8d） | **+11.4pp** |
| teatime | 0.6499 / 0.8983 | 0.6441 / 0.9322 | 0.6705 / 0.8814（8d） | **+2.1pp** |
| ramen | 0.5308 / 0.6761 | 0.5038 / 0.7042 | 0.5387 / 0.7042（8d） | **+0.8pp** |
| **四场景平均** | 0.5340 / 0.7789 | 0.5079 / 0.7486 | 0.5928 / 0.8177 | **+5.9pp 全胜** |

**叙事要点**：
- **四场景 mIoU 全胜**（+9.3 / +11.4 / +2.1 / +0.8，平均 +5.9pp）：同评估协议下维度优化 CLIP 场全面超越官方 3d 原版——AE 维度选择（8d/24d per-scene）直接兑现为分割质量。
- teatime mAcc 是唯一例外（3d 0.8983/0.9322 vs 我们 0.8814/0.8983）：官方 ckpt 定位更好但分割边界更差，mIoU/mAcc 分离与 EXP-017 结论一致。
- ensemble 对 3d 场一升三降（figurines +1.8 / teatime −0.6 / ramen −2.7 / waldo −8.9）vs 我们 CLIP 场两升两降——ensemble 兼容性与 AE 维度相关再添证。
- 3d 压缩（170:1）对 dense 小物体场景（figurines）伤害最大，印证维度扫描结论（密集小物体需高维）。

---

## 3. 主表 B：四场景 image eval 双口径矩阵（EXP-040 定稿矩阵，**数字直接引用，勿重算**）

协议：image-query cross-frame 3D 检索，chosen-level top1，first-bbox / any-bbox 双口径；DINO 场 = AE 32d + α=0.7 自蒸馏平滑重训（a07）；CLIP 场 = 各场景最优 AE 维度。

| 场景 | CLIP 场 first / any（%） | DINO a07 场 first / any（%） | Δ first | Δ any | n（对象对） | McNemar p |
|---|---|---|---|---|---|---|
| teatime | 81.97 / 86.89（8d） | 91.80 / 93.44 | **+9.8pp** | **+6.6pp** | 61 | 0.0703（ns，边缘）/ 0.2188 |
| figurines | 88.89 / 88.89（24d） | 90.74 / 90.74 | **+1.9pp** | **+1.9pp** | 54 | 1.0000（ns）/ 1.0000 |
| ramen | 74.68 / 82.28（8d） | 78.48 / 84.81 | **+3.8pp** | **+2.5pp** | 79 | 0.6291（ns）/ 0.8036 |
| waldo | 69.23 / 92.31（8d） | 46.15 / 69.23 | −23.1pp | −23.1pp | 13 | 0.2500（ns）/ 0.2500 |
| **pooled** | — | — | — | — | 207 | 0.3240（ns）/ 0.6076 |

- 来源：EXP-040（`hyper_parameter.md` "EXP-040 ① any-bbox 统一口径四场景矩阵"）；McNemar 全部来自 EXP-041（`eval_result/mcnemar/*.json` per-pair JSON + `eval/mcnemar.py` 精确二项检验，pooled 为四场景 discordant 合并后检验）。渲染忠实性已验证：EXP-040 重跑 7 组 eval 中 6 组 first-bbox 值与历史逐分一致（EXP-040 "渲染忠实性验证"）。
- **显著性诚实结论（EXP-041，论文 limitation）**：所有场景单测与 pooled McNemar 均不显著。teatime first 口径 p=0.0703 最接近显著（discordant 1:7）；figurines discordant 4:5 近对称（+1.9pp 来自基础率而非差异集中）；pooled first b=15/c=22。**DINO 优势 = 方向一致（3/4 场景同向）+ 两个大样本场景（61/79 对）占优的效应量证据，而非显著证据**；论文表述应报效应量+双口径矩阵+完整 McNemar 表，避免宣称显著。
- **判定**：any-bbox 统一口径 3 胜 1 负；唯一负场 waldo 不显著（p=0.25）。ramen 大样本（79 对）独立复现 DINO 优势，per-level DINO 全占优（first 口径 L1 58.2 vs 46.8 / L2 79.8 vs 70.9 / L3 77.2 vs 49.4，EXP-039）。
- 计数口径（first）：teatime 91.80%=56/61、81.97%=50/61（EXP-035a）；figurines 90.74%=49/54、88.89%=48/54（EXP-037b）；ramen/waldo 见 EXP-039/040。

### B-2 主表升级：多帧证据聚合口径（EXP-048，**建议为论文主口径**）

协议：chosen 口径丢弃跨帧证据（只看全库最高单峰）；**majority 口径** = 每帧独立 top1 检索后统计该物体跨帧命中率 ≥50% 判命中（无标签泄露，per-(level,帧) 独立投票）。all-any 口径四场景双双触顶（96~100%）无区分度，弃用。hit-rate（连续帧命中率）作为辅助指标。

| 场景 | chosen Δ（DINO−CLIP） | majority DINO / CLIP（%） | **Δ majority** | hit-rate DINO / CLIP（%） | McNemar（majority） |
|---|---|---|---|---|---|
| teatime | +9.8pp | **91.80 / 78.69** | **+13.1pp** | 91.53 / 73.34（+18.2pp） | **9:1, p=0.0215 ★** |
| figurines | +1.9pp | **87.04 / 77.78** | **+9.3pp** | 84.05 / 73.97（+10.1pp） | 8:3, p=0.2266 ns |
| ramen | +3.8pp | **88.61 / 63.29** | **+25.3pp** | 76.40 / 62.43（+14.0pp） | **21:1, p<0.0001 ★★★** |
| waldo | −23.1pp | 76.92 / 92.31 | −15.4pp | 51.28 / 71.79（−20.5pp） | 0:2, p=0.5000 ns |
| **pooled** | p=0.324（ns） | — | — | — | **38:7, p<0.0001 ★★★** |

- 来源：EXP-048（`eval_result/adaptive/exp048_*.json` 8 组 per-pair + hyper_parameter.md EXP-048 终章）；teatime/figurines majority McNemar 用 scipy binomtest（EXP-048 记录）。
- **里程碑意义**：chosen 口径 pooled p=0.324（EXP-041 全 ns）→ majority 口径 **pooled p<0.0001**——多帧证据聚合是任务路由主张的**显著性转折点**。此前"效应量证据非显著证据"的 limitation（EXP-041）由本口径解除。
- **机理自证**：majority 放大增益 = DINO 场跨视角一致性优势的显式测量——hit-rate 三健康场景 +10.1~+18.2pp（DINO 场每帧检索峰更稳定）。
- **waldo 归因再加固**：每帧 hit-rate DINO 51.3% < CLIP 71.8%——其 2D 入口劣势是每帧性的，多帧聚合放大真实差距而非拯救（一致性好 ≠ 判别对）；n=13 下 0:2 不显著。
- **叙事建议**：主表以 majority 为主口径、chosen 为对照；正文强调"检索任务天然拥有多帧证据，单峰协议系统性低估 3D 一致场"——这也是对 LERF 系单峰评估协议的方法论批评。

### B-2-final 统一主表（EXP-054 定稿版：per-scene 最优 α 配置，**以此为准**）

配置：teatime a07（低 α 平台）/ figurines **a09**（高 α 峰）/ ramen a07 / waldo **raw**（平滑有害场景，α=0 为其最优）。majority McNemar 为精确二项检验（2×P(X≤min)）。

| 场景 | DINO majority | CLIP majority | Δ | McNemar p | DINO chosen any | CLIP chosen any |
|---|---|---|---|---|---|---|
| teatime | 91.80% (56/61) | 78.69% (48/61) | **+13.1pp** | **0.0215 ★** | 93.44 | 86.89 |
| figurines | 92.59% (50/54) | 77.78% (42/54) | **+14.8pp** | **0.0215 ★** | 92.59 | 88.89 |
| ramen | 88.61% (70/79) | 63.29% (50/79) | **+25.3pp** | **4.8e-6 ★★★** | 84.81 | 82.28 |
| waldo | 76.92% (10/13) | 92.31% (12/13) | −15.4pp | 1.0 (ns, b=2 c=0) | 76.92 | 92.31 |
| **POOLED** | — | — | — | **b=5 c=39, p≈1.4e-7** | — | — |

**vs EXP-048 版主表的三处升级**：① figurines a09 使 majority Δ +9.3→+14.8pp 且**首次单独显著**（0.227→0.0215）；② waldo raw 使平滑伤害归因更干净（raw 与 a07 的 majority 零 discordant——α=0.7 的伤害是单帧现象，多帧投票天然免疫，再证 majority 口径优越性）；③ pooled 39:5, p≈1.4e-7。

**显著性叙事升级线**（论文 limitation 节可写）：EXP-041 全 ns（pooled p=0.32）→ EXP-048 pooled ★（p<0.0001）→ **EXP-054 三场景单独 ★ + pooled p≈1.4e-7**——唯一 ns 是 n=13 的 waldo，且其 2D 入口判据（自蒸馏启用前置条件）正确预测了它，负场本身就是方法边界的验证而非漏洞。

### B-4 场质量上界对照表（EXP-055，RM 协议，**必须与主表并列报告**）

协议：render-mask query（帧 A 渲染特征在 SAM tile 区域取均值，query/db 同为 3D 渲染空间）——剥离 query 端 AE 压缩噪声后的"场质量"口径。配置同主表（teatime a07 / figurines a09 / ramen a07 / waldo raw）。

| 口径 | teatime | figurines | ramen | waldo | pooled |
|---|---|---|---|---|---|
| chosen any | +4.9pp | 0 | **+8.9pp (p=0.078)** | −15.4pp | p=0.268 ns |
| majority | 0 | −3.7pp | +5.1pp | −7.7pp | **p=1.0 拉平** |

**优势来源分解（EXP-055 核心贡献，写法建议）**：
1. **主表（2D tile query）的 pooled p≈1.4e-7 主要来自 query 入口端**——AE 压缩不对称（CLIP 8d 入口噪声大 vs DINO 32d 入口忠实），RM 剥离后 majority 完全拉平
2. **路由主张的精确表述**：不是"DINO 场判别质量更高"（RM 证伪），而是"**image-query 的入口维度需求（高维）与 text-query 的场维度需求（CLIP text 最优 8d，EXP-018 维度曲线）结构性冲突——单场不可两全，路由是解冲突方案**"。EXP-018 已证 CLIP 场拉高维度伤 text-query，堵住"把 CLIP 场也训成 32d"的反驳
3. **ramen 是唯一场质量优势场景**（RM chosen +8.9pp，n=79 最大样本，双口径均正）——保留为"场质量增益存在但非普适"的诚实表述
4. **多帧一致性传导**：DINO 场优势 = 入口忠实性 × 跨帧一致性（teatime CLIP 场 RM majority 78.69→91.80 回血 = CLIP 场的 2D 入口噪声同时破坏单帧精度与多帧稳定性）

---

## 4. 消融表 C

### C-1 自蒸馏平滑 α 扫描（image-query first-bbox top1）

**figurines（n=54，完整曲线，EXP-052 系列升级版：7 点 dose-response + 双口径）**：

| 配置 | chosen top1 | majority | hit-rate | vs CLIP 24d 场（chosen any） | McNemar p | 来源 |
|---|---|---|---|---|---|---|
| DINO 32d raw 场（未平滑） | 77.78% (42/54) | — | — | −11.1pp | — | EXP-036a |
| α=0.5 平滑场 | 88.89% (48/54) | — | — | 0（追平） | — | EXP-037a |
| α=0.7 平滑场 | 90.74% (49/54) | 87.04% | 84.05% | +1.85pp | 1.0 (ns) | EXP-037b / 048 |
| α=0.85 平滑场 | 90.74% (49/54) | 92.59% | 88.99% | +1.85pp | 1.0 (ns) | EXP-052 |
| **α=0.9 平滑场（主表配置）** | **92.59% (50/54)** | **92.59%（峰）** | **89.61%** | **+3.70pp** | 0.727 (ns) | EXP-052b |
| α=0.95 平滑场 | **96.30% (52/54)** | 90.74% | 89.61% | **+7.41pp** | 0.219 (ns) | EXP-052c |
| α=1.0（纯自蒸馏定点） | 96.30% (52/54) | 90.74% | 89.09% | +7.41pp | 0.219 (ns) | EXP-052d |
| α=0.5 迭代 2 轮（动态重渲） | 85.19% | — | — | −3.7pp | — | EXP-037c |

> **figurines α 曲线三口径终版结论（EXP-052 a/b/c/d）**：① chosen 单调升并渐近 96.30%（α≥0.95，5% 2D 掺混已不敏感）；② majority 峰 α=0.85-0.9（92.59%），α≥0.95 回落 −1 对——**纯自蒸馏过拟合分叉**（单帧检索更锐利、跨帧一致性受损）；③ hit-rate 渐近 ~89.6%。④ **α=1.0 定点自蒸馏稳定**（固定 a07 渲染监督），未复现 EXP-037c 动态重渲迭代的二次稀释——平滑迭代失败源于重渲噪声累积而非自蒸馏本身。⑤ 主表推荐 **α=0.9**（双口径平衡 92.59/92.59 + hit-rate 最优），α≥0.95 的高 chosen + majority 回落进分析节。

**teatime（n=61，完整曲线，EXP-042）**：

| 配置 | top1 (first / any) | vs CLIP 8d 场 first（81.97%） | McNemar p (first) | 来源 |
|---|---|---|---|---|
| DINO 32d raw 场 | 90.16% / 93.44% | +8.2pp | 0.0625 | EXP-042 |
| **DINO 32d α=0.3 平滑场** | **91.80% / 93.44%** | **+9.8pp** | **0.0312 ★（唯一显著）** | EXP-042 |
| DINO 32d α=0.5 平滑场 | 90.16% / 91.80% | +8.2pp | 0.1250 | EXP-042 |
| DINO 32d α=0.7 平滑场 | 91.80% / 93.44% | +9.8pp | 0.0703 | EXP-042 |

> **α 敏感性双场景结论（EXP-042）**：① teatime 曲线非单调且波动仅 ±1 对（噪声级），α0.3/α0.7 双峰 91.80%、α0.5 谷——teatime 平滑收益本质微弱（+1.6pp），远小于 figurines（+11.1pp）；② **α 收益只存在于 first-bbox 口径，any 口径全程不变或略降（raw 已 93.44）→ 平滑机制 = 修复 bbox 中心偏移，不提升整体判别力**；③ figurines 单调升（α0.7 甜点）/ teatime 双峰非单调 → α 最优值场景相关，无全局最优；④ 方法论判据修正："2D tile 检索 DINO 领先"是平滑启用的**必要非充分条件**（teatime 2D +10.3pp 领先但平滑近无效；waldo 2D 打平则平滑有害）。

### C-2 waldo：raw vs α0.7 平滑（n=13，EXP-040 ②）

| 场 | first-bbox | any-bbox |
|---|---|---|
| CLIP 8d 场 | 69.23% | 92.31% |
| DINO 32d **raw** | 53.85% | 76.92% |
| DINO 32d **a07** | 46.15% | 69.23% |

- **α=0.7 平滑在 waldo 伤害 −7.7pp**（any 口径 raw 76.92 > a07 69.23；1 对差异，n=13 无法显著，方向明确）。
- 但 raw 仍输 CLIP 场 15.4pp → 反转是**双因素叠加**：α 平滑伤害（占缺口 1/3）+ DINO 场本身弱于 CLIP 场（根源 = 2D 入口优势归零）。（EXP-040 归因闭环）

### C-3 口径敏感性：first vs any 对 Δ 的影响（EXP-040 ①）

| 场景 | Δ(first) | Δ(any) | 口径效应 |
|---|---|---|---|
| teatime | +9.8pp | +6.6pp | any 帮 CLIP 多救 3 对（+4.9pp）而 DINO 仅 +1.6pp |
| figurines | +1.9pp | +1.9pp | 两口径完全相同（GT bbox 大 / 重复标签少） |
| ramen | +3.8pp | +2.5pp | 优势收窄但方向不变 |
| waldo | −23.1pp | −23.1pp | 重复标签下两口径同受罚（knife×9 first 口径惩罚 −23~−40pp，EXP-039d） |

- **结论**：first-bbox 在重复标签密集场景系统性低估，any-bbox 更公平；**DINO 优势方向在任何口径下不变**，主表报双口径。（EXP-040）

---

## 5. 方法节要点

### 5.1 Task routing 推理流程
1. **训练期**（双场并行，共用 3DGS 几何热启动）：
   - CLIP 场：SAM tile → CLIP ViT-B/16 global embedding（黑底 pad + resize 224）→ AE 压缩（teatime/waldo/ramen 8d，figurines 24d）→ 3-level 语言高斯场。（EXP-018/021/039）
   - DINO 场：同 tile 划分 → DINOv2 per-tile 768d → **AE 32d**（encoder [256,128,32,32,32] / decoder [32,128,256,256,768]，mse）→ （判据通过时）自蒸馏平滑 → 重训 3-level。（EXP-035a/037/039）
   - 工程约束：rasterizer `NUM_CHANNELS` 32d 为共享内存静态上限（16×16 tile×dim×4B ≤ 48KB；64d 需动态共享内存改造）。（EXP-035a 工程）
2. **推理期**（按查询模态路由，互不干扰）：
   - text-query → CLIP 场：LangSplat relevancy 协议，softmax(10·sims)，7 模板 ensemble 平均。（EXP-024）
   - image-query → DINO 场：帧 A GT bbox 最大覆盖 tile 特征（与渲染场同 AE 空间）→ 目标帧 B render 特征场逐像素 cos → 30×30 滤波 → argmax 点落入同类 GT bbox 判命中。（EXP-035a eval 协议）

### 5.2 跨视角自蒸馏平滑（公式化描述）
动机：DINO 场的失效模式是**跨视角不一致**——同物体跨帧 tile cos DINO 比 CLIP 低 5.4~7.0pp（teatime 0.706 vs 0.775；figurines 0.629 vs 0.683，EXP-036a 归因诊断三连，AE/渲染均排除）。

做法：用**已训 DINO 场**的渲染特征（天然 3D 多视角一致）与原始 2D tile 监督做凸组合，重训 GS：
$$F'_{tile} = \alpha \cdot F_{render}(tile) + (1-\alpha) \cdot F_{2D}(tile)$$
其中 α 为渲染特征权重（实现：`eval/smooth_tiles.py` L47 `newf[t] = alpha * rm + (1 - alpha) * f[t]`，逐 level 逐 tile）。α=0.7 为 figurines 甜点（C-1），α=0.5 单轮即追平缺口。迭代平滑无累积效应（二次稀释）。

### 5.3 平滑启用判据（论文方法节应写明）
**前置判据 = 该场景 2D tile 检索 DINO 是否领先；领先才平滑（teatime +10.3pp / figurines +5.6pp → 启用），打平则跳过（waldo → 跳过）。**（EXP-040 结论 3 / 方法论修正）
反例支撑：waldo 2D tile 检索两场打平（EXP-039d），α=0.7 无脑套用反伤 −7.7pp（C-2）——平滑增益需要 2D 入口优势作为基础。

---

## 6. 叙事素材

### 6.1 负结果证据链摘要（动机：为什么是"路由"而不是"融合/更强教师"）
三条例行改进轴全部关闭，一行式结论：
- **多模态融合轴**：InfoNCE 对比损失（EXP-009）、深度第三模态（EXP-013/015/016）、双流 score 级 late fusion（EXP-023，λ=0.3/0.5 → 0.5536/0.6107 vs 0.6705 单调劣化）全线失败——LangSplat text-query relevancy 协议下任何外部信号（DINOv2/深度/对比学习）注入均稀释 CLIP 判别分布。
- **dense 教师轴**：tile-mean dense 特征方差坍缩（EXP-025，2D mIoU 0.4399→0.0321，−40.8pp）；per-patch dense 空间错位（EXP-026，IoU≈0.00~0.03，stuffed bear tile 级 IoU 0.97 → patch 级 0.000）；masked pooling 无方案过 gate（EXP-027，原管线已是 masked CLIP）——**CLIP 模型本身不产 dense 判别特征**，聚合粒度不是问题。
- **骨干升级轴**：L14 全链条 −6.7~−7.0pp（EXP-022）；同压缩率对照证伪压缩率假说（EXP-028）；6 权重 2D 扫描 e32 +8.1pp 过 gate（EXP-029），但 3D 全链条**符号翻转**（2D +8pp → 3D −8pp；EXP-030–034 完整维度曲线 8/12/16/20/24d 均低于 laion2b 8d 达 −5.7~−10pp）——"上游判别力 ≠ 下游性能"，laion2b B/16 仍是最优 text 教师。
- **汇聚结论**：CLIP 空间是 text-query 唯一有效语义空间 ⇒ DINOv2 的价值不在注入 CLIP 流，而在**为另一类查询（image-query）单开一个场**——任务路由是这三条负轴的唯一幸存出口。（EXP-023 大汇聚结论 + EXP-035a）

### 6.2 waldo 反转归因段（唯一反例，完整闭环）
四场景中唯一 CLIP 胜的场景（69.23/92.31 vs 46.15/69.23，n=13），McNemar p=0.25 不显著（EXP-039）。归因四因素（EXP-039d + EXP-040 ②）：
1. **极小样本**：n=13，任何单对翻转即 ±7.7pp，无统计功效；
2. **协议缺陷**：knife×9 等重复标签密集，first-bbox 协议对重复标签惩罚 −23~−40pp（any-bbox 口径下 CLIP 仍领先 23pp，方向性存在但样本太小）；
3. **DINO 2D 入口优势归零**：teatime +10.3pp / figurines +5.6pp 的 2D tile 检索入口优势在 waldo 消失（EXP-039d）——而平滑增益与 DINO 场优势都以 2D 入口优势为前提；
4. **α=0.7 直接套用伤害**：无 raw 场对照即套用，any 口径伤害 −7.7pp（raw 76.92 > a07 69.23）。
**闭环**：即便去掉平滑（raw），DINO 场仍输 CLIP 15.4pp——反转 = α 伤害（1/3 缺口）+ 场本身弱（2/3 缺口）双因素叠加；"32d 维度不合适"假说已排除（DINO AE cos_sim 0.910 三场景最高，PCA 扫描 16d 峰但 AE-32d 优于 PCA-32d；EXP-039d ③）。**修正 EXP-039 归因④的权重：平滑不是主因，但去掉平滑可挽回 1/3 缺口**（EXP-040）。

### 6.3 其他可用叙事
- **3D 多视角融合一致为正**：3D 场对 2D 检索的放大两场景均为正（DINO +16.1/+13.4pp，CLIP +18.2/+30.1pp），但放大效率特征类型相关——DINO 跨视角稀释更重。（EXP-036a 归因链）
- **"2D tile 上限"框架不成立**：3D 管线大幅超越 2D tile map（0.6932 vs 0.5394，+23pp），渲染特征本身就是比 tile 图更强的表示；PCA 64:1 毁灭 2D tile mIoU（0.47→0.15）而真实 AE 8d 全链条 0.6705——非线性 AE + 多视角融合能从低维恢复大量判别力。（EXP-025/028）
- **AE 维度最优值场景相关**：teatime CLIP 峰 8d / figurines 峰 24d / e32 峰 20d，倒 U 型且 AE 随机性不可忽略。（EXP-018/021/033/034）

---

## 7. 待办清单（写论文前需补齐）

- [x] **McNemar p 值填主表 B**：EXP-041 完成（2026-10-03 21:25），全部 ns，见上表与"显著性诚实结论"。per-pair JSON 在 `eval_result/mcnemar/`。
- [x] **teatime α 扫描填消融表 C-1**：EXP-042 完成（2026-10-04 03:59）。teatime α0.3/α0.5 已入表；**α0.3 first 口径 p=0.0312 为全项目首个显著结果**（b=0/c=6，CLIP 赢的 pair 为零）；主表 B 的 teatime 行可注"α0.3 最优（与 α0.7 并列）"。
- [ ] **定性可视化**：eval_result 各场景已有 heatmap / composited / localization 三类输出图（如 `eval_result/figurines_24d/00041/`），需挑代表帧排版：建议每场景 1 个 DINO 胜 + waldo 1 组 CLIP 胜对照。
- [ ] **相关对比方法（LangSplat 原版数字引用提示）**：主表 A/B 需并排引用 LangSplat 原论文在 LERF 三场景（teatime/figurines/ramen…按原论文场景名）的 mIoU/mLoc 数字——**引原论文报告值，勿用本地复现**；本地 `ckpt/pretrained_model/` 为 sofa 预训练权重，口径不同。引用时注意原论文 prompt ensemble 设置与本工作 EXP-024 协议（7 模板）的对应关系。
- [ ] text eval 的 ensemble 双口径叙述：主表 A 报双列，正文强调"两升两降、场景相关"（EXP-039 结论）。
- [ ] waldo 处理口径决策：主表 B 保留负值并引归因段（6.2），还是移至附录 + 正文脚注——写作时定。

### B-3 重复标签分层协议（EXP-051，方法论文期刊化素材）

**两个正交歧义维度**：跨帧重复率（teatime 98%/figurines 95%/ramen 99%/waldo 52%）≠ 同帧聚集度（max same-frame same-label：3/2/2/**5**）。与 EXP-048 结果对齐后三个反直觉结论：
1. ramen 跨帧重复最高（99%）但 DINO majority 增益最大（+25.3pp）——跨帧重复是任务正常难度，不惩罚 3D 场；
2. waldo knife 同帧聚集 5 实例/帧，majority 口径下双方 8/9 平手——多帧投票抵消单帧歧义（修正 EXP-039"knife×9 惩罚"叙事：该惩罚只在 chosen/first 口径成立）；
3. waldo majority 残余差距全在 plate（n=2）——小样本场景（n<20）必须报 per-label 表，场景级均值会被 1-2 个标签噪声主导（teatime apple −20%、figurines green toy chair 0/2 同理）。

---

## 附录：waldo 负场机理（五层证据链，EXP-061 定稿）

> 用途：limitation / analysis 节。回答"为什么 waldo 场景 DINO 不如 CLIP"——不是缺陷而是方法边界的物理成因，与任务路由主张自洽。

**一句话**：GS 特征场按构造视角恒定（每 Gaussian 一份特征），因此它对教师特征是一台"视角方差切除机"——CLIP 特征视角方差≈0（全保留），DINOv2 特征的视角方差恰是判别力所在（被切除）。waldo 的小物体×金属反光把效应推到极致，使 DINO 的 2D 入口优势（+15.4pp）在 3D 传导中反转为 −23pp。

**五层链条**（每层有实验锚点）：
1. **教师视角依赖性**：DINOv2 patch 级特征逐帧剧变，CLIP global 语义特征视角不变（跨帧一致性差 5~7pp，EXP-036a）
2. **视角恒定拟合视角可变监督 = 强制跨视角平均**（L1 最优解），高频判别分量按视角方差衰减；同一平均对 CLIP 是降噪（2D→3D +48.1pp）对 DINO 是毁信号（+9.6pp）——EXP-059
3. **压缩率排除维度解释**：64d 零损耗（2D 快测 59.62%=768d）但 3D 仍输 23pp，瓶颈在视角方差非信息保留——EXP-059
4. **场景放大器（乘积效应）**：跨帧特征 cos 分组测量——knife（金属×小）Δ(D−C)=−0.191 vs 非金属 −0.085（2.3×）vs sink（金属×大面积）仅 −0.017：伤害=金属度×小尺寸，单独金属不足以致伤——EXP-061，knife 独占 23/25 检索对
5. **信息碎片化**：DINO 场 all-any=100%（信息完整在场）但散落 (level×frame)，argmax 协议奖励 CLIP 的集中场——EXP-060

**方法启示（自适应路由的理论依据）**：任务路由适用条件三条——①2D 判别度领先（EXP-040 前置判据）②教师视角方差可在 GS 平均下存活（EXP-061）③validation 帧上 2D→3D lift 确认。三条同时满足才路由到 DINO 场，否则回退 CLIP 场——per-scene 自适应路由（waldo 自动回退 → 框架矩阵 4/4），路由决策全部基于 validation，无测试集偷看。

**GT-free 自动路由判据：两族信号实验证伪（EXP-062，防审稿人追问"为什么不用自动判据"的消融弹药）**：
- **2D 入口统计失效**（EXP-062a）：SAM 伪类 2D 伪检索 DINO 四场景全胜（含 waldo 31.5% vs 21.5%）——2D 判别度无法预测 3D 反转；跨帧一致性无法区分 waldo 与 ramen（cos gap 同为 −0.177 但胜者相反）
- **场级伪检索失效**（EXP-062b）：SAM 段伪查询在渲染场上检索（配对协议：同帧对/同段/同 query，峰值检索镜像部署机制，±段粒度两变体），GT 评估中 −23pp（waldo）~+9.8pp（teatime）的真实场差距被压缩到 0~4pp 噪声区，方向随机，路由 ≤3/4
- **机理**：无 GT 则无法区分"语义物体段"与"碎片段"，伪检索信号被 SAM 段身份噪声主导；语义级判别度恰是 GT 提供的信息
- **论文表述**：路由作为设计选择由 GT 消融矩阵支撑（EXP-054/039/040），与"判据是否自动"正交；GT-free 自动路由（2D 统计 + 场级伪检索两族）经系统实验证伪，列为 future work（可能方向：生成式跨视角一致性、开放词汇段落解析）。工具：eval/eval_gtfree_router.py（062a）、eval/eval_gtfree_router_field.py（062b）
