# LangSplat 任务路由 — SOTA 调研与投稿策略

> 整理日期：2026-10-07。基于：① 联网检索（arXiv 2603.24146 / 2606.30638 / 2606.19733 / 2410.07577 / 2406.02058 / 2403.15624 / 2512.20927 等）；② 内部实验记录 `hyper_parameter.md`（EXP-001~053）与 `paper_materials.md`。
> **协议警示（全文最重要的一个事实）**：LERF-OVS 的 mIoU 在不同论文间**不可直接横比**——同一方法（LangSplat）在不同论文的重评测表中从 ~0.53 到 0.92+ 都有报告（mask 阈值、prompt 集、GT 处理、评测帧选择均未统一）。这既是本调研的难点，**也正是我们协议贡献（EXP-048/051/053）的动机**。

---

## 1. 任务 Landscape：三条评测线

| 评测线 | 定义 | 现有 SOTA 竞争强度 | 我们的参与度 |
|---|---|---|---|
| A. text-query 开放词汇分割 | 文本 → 2D 渲染 mask，mIoU | **极度拥挤**：LangSplat → Feature-3DGS → LEGaussians → IEGaS(ICLR'25) → GaussDet(2026)，每代 +1~3pp | 弱（CLIP 场即 LangSplat 协议复现+微调，无新方法主张） |
| B. 3D 定位/点级理解 | text/image → 3D 位置 | 中等：LERF → OpenGaussian(NeurIPS'24, 点级) | 中（Localization accuracy 一直作为辅助指标） |
| C. **image-query 跨帧 3D 检索** | 图像区域 → 跨帧 3D 场检索同实例 | **近乎空白**：QueryGaussian(2026) 做 text-query 实例检索；Ilov3Splat(2026) 相邻；无人以"查询模态路由"视角系统研究 | **我们的主场**（EXP-035~053 全链自建协议） |

**战略结论**：线 A 不要正面拼 SOTA（每个代际都有专职团队，且我们无方法级增益）；论文应定位为**线 C 的任务定义 + 协议 + 系统分析**，线 A 作为"text-query 侧不退步"的约束条件报告。

---

## 2. SOTA 性能盘点

### 2.1 线 A：LERF-OVS text-query mIoU（已发表论文报告值，协议各异）

| 方法 | 发表 | 报告表现（协议自定，不可横比） | 备注 |
|---|---|---|---|
| LERF | ICCV'23 | mean mIoU ≈ 0.40 级 | 慢（~30min/场景），基线 |
| LangSplat | CVPR'24 | 原文报告显著超 LERF；IEGaS 重评测表中单场景 mIoU/mAcc 达 92.5/94.2 | 我们的直接基线；不同论文对其复测值差异 0.53~0.92+ |
| Feature-3DGS | CVPR/SIGGRAPH Asia'24 | IEGaS 表：83.5~93.4（per-scene mIoU） | 免 CLIP 蒸馏、快（~11fps 推理） |
| LEGaussians | CVPR'24 | IEGaS 表：84.9~92.5 | 语言嵌入+不确定性 |
| **IEGaS (3D Vision-Language GS)** | **ICLR'25** | 自报 SOTA：开放词汇分割显著超上述全部 | 当前线 A 事实 SOTA；跨模态 rasterizer + 视角混合 |
| OpenGaussian | NeurIPS'24 | 点级 3D 开放词汇（选择/点击/点云理解） | 不报 LERF-OVS mIoU 主表，报点级任务 |
| GaussDet | 2026 (arXiv 2606.30638) | LeRF-OVS + ScanNet 开放词汇分割 + referring grounding（零样本 +16.7% mIoU） | 2D 检测器路线，正面对拼线 A |
| LightSplat | 2026 (arXiv 2603.24146) | 5 秒级开放词汇理解；在更难的 DL3DV-OVS 上各方法 mIoU 大幅走低（LangSplat 系 ~8-15 级） | 新 harder benchmark 趋势 |
| Quantile Rendering | 2025 (arXiv 2512.20927) | 高维特征直接嵌入 3DGS（绕开 AE） | **与我们 AE 32d 压缩瓶颈结论直接对话** |

**判读**：线 A 的 SOTA 已被 ICLR'25/GaussDed 推到很高水平，且社区正转向更大数据集（DL3DV-OVS/ScanNet）。我们 teatime 0.6932（自有协议）无跨论文可比性——**不要把线 A 数字当卖点**。

### 2.2 线 C：image-query 跨帧 3D 检索（我们的协议，无外部可比数字）

| 场景 (n) | CLIP 场 chosen/any | DINO a07 场 chosen/any | Δ chosen | majority McNemar |
|---|---|---|---|---|
| teatime (61) | 81.97 / 86.89 | 91.80 / 93.44 | +9.8pp | 9:1 **p=0.0215** |
| figurines (54) | 88.89 / 88.89 | 90.74 / 90.74 | +1.9pp | 8:3 ns |
| ramen (79) | 74.68 / 82.28 | 78.48 / 84.81 | +3.8pp | 21:1 **p<0.0001** |
| waldo (13) | 69.23 / 92.31 | 46.15 / 69.23 | −23.1pp | 0:2 ns |
| **pooled (207)** | — | — | — | **38:7 p<0.0001** |

端到端协议升级（EXP-053 render-mask query）：3胜1负结构不变，ramen +10.1pp（p=0.0215），pooled 16:6 p=0.0525。

**这块是论文的核心资产：任务定义新颖 + 协议完整（3 query 协议 × 4 口径 × McNemar × per-label 分层）+ 显著性闭环。**

### 2.3 近邻工作差异声明（审稿人最可能引用的 5 篇）

| 近邻 | 重叠点 | 我们的差异（写进 related work 划界） |
|---|---|---|
| LERF | 双场先例（CLIP+DINO relevancy） | LERF 的 DINO 场只做 relevance 辅助，从不做检索/定位主任务；我们按**查询模态路由**并系统证明 |
| OpenGaussian | SAM tile + 3D 一致实例特征 | 它服务 text/click query（CLIP 关联）；我们服务 image query 且给出"何时该用 DINO 场"判据 |
| QueryGaussian (2026) | 3D 实例检索任务名近似 | 它是 **text**-query、training-free、按需 2D 提升；我们是 **image**-query、场内检索、协议+路由 |
| seconGS (2503.21767) | mask 跟踪重蒸馏修一致性（与 α 平滑动机近） | 它修 SAM mask 时序对应；我们混回 2D 特征本身；且我们给出 6 条替代路线全败的对照证明 |
| Quantile Rendering | 特征嵌入 3DGS 的替代方案 | 它绕开 AE 塞高维特征；我们的压缩瓶颈证据链（EXP-028/030/053）恰好说明这是对的方向，可在 discussion 互引 |

---

## 3. 我们结果的优势（放大什么）

| # | 优势 | 证据 | 为什么审稿人会喜欢 |
|---|---|---|---|
| S1 | **显著性闭环**：majority 口径 pooled p<0.0001（38:7），ramen 单场景 p<0.0001 | EXP-048 | 大量 3D 语义论文只报均值不报检验；我们有 per-pair JSON + 精确二项检验 |
| S2 | **协议方法论贡献**：证明单峰协议系统性丢弃跨帧证据、query 端 AE 压缩不对称（CLIP 8d 64:1） | EXP-048/053 | 对全社区有效的可复用结论，命中率高于"又涨 1pp" |
| S3 | **可证伪的机制结论**："增益唯一来自 2D 信息回灌"有 6 条独立失败路线对照（门控/正则/结构蒸馏×3侧/推理端×2） | EXP-043/044/045/046/049/050 | 审稿人最爱问"为什么不用 X"——我们全试过且知道为什么不行 |
| S4 | **路由判据可操作**：2D tile 快测（30 分钟免训练）预测 3D 检索优势 | EXP-035a/039/040 | 实用性：给别人什么时候该建第二场的决策规则 |
| S5 | **诚实评估文化**：负场景 waldo 完整归因（knife 同帧聚集 + plate n=2）+ 四口径全报 | EXP-039/041/051 | benchmark 论文的可信度来源 |
| S6 | **text-query 侧零成本**：路由不伤 CLIP 场（LangSplat 协议保持） | 主表 A | 消除"伤一刀"疑虑 |

## 4. 劣势与风险（缩小什么）

| # | 劣势 | 风险等级 | 缓解策略（写作层面 / 实验层面） |
|---|---|---|---|
| W1 | **单数据集 4 场景、n 小**（13~79 对） | 高 | 写作：全部报效应量+检验+per-label；实验：补 3D-OVS 2-3 场景（方向b，每场景 ~1 天） |
| W2 | **无第三方基线在线 C** | 高 | 写作：定位为"任务定义+协议"论文（新任务无先例基线是合理起点）；实验：把 OpenGaussian/QueryGaussian 适配到 image-query 协议（工程量中-大，可作 rebuttal 弹药或 camera-ready） |
| W3 | **自比自**（我们的 CLIP 场 vs 我们的 DINO 场） | 高 | 写作：明确说 CLIP 场=LangSplat 原协议代表；"路由增量"定义清楚；实验（低成本版）：加 raw DINO 场消融已有（EXP-042 α 曲线就是） |
| W4 | **线 A 数字与文献不可比** | 中 | 写作：全文只用自有协议并明示；实验：跑一次 LangSplat 官方 eval 脚本得一行 apples-to-apples 数字（~2h） |
| W5 | **waldo 负场景** | 中 | 写作：转为"边界条件刻画"——2D 入口判据正确预测了它（DINO 2D 打平→不启用平滑）；per-label 显示 knife 8/9 平手 |
| W6 | **方法组件简单**（双场+凸组合平滑） | 中 | 写作：简单=可复现+正交于骨干改进；把深度放在分析节（S3 证据链）；标题避免"novel architecture"式词汇 |
| W7 | AE 32d 是工程上限（rasterizer 共享内存） | 低 | 写作：作为 limitation + 未来工作（动态共享内存/Quantile Rendering 路线互引） |

---

## 5. 论文写作策略

### 5.1 叙事主线（一段话版）
> 开放词汇 3D 理解的查询模态天然异构。我们证明：与其把 text/image 查询挤进单一 CLIP 对齐场（30+ 实验的融合/蒸馏/正则路线全部失败），不如按"教师判别空间与查询模态匹配度"路由到独立特征场；image-query 侧 DINO 场在多帧证据口径下 pooled p<0.0001。我们同时贡献：揭示单峰检索协议系统性低估 3D 一致场的评估方法论、query 端压缩不对称的量化分析、以及 6 条替代路线的完整失败图谱。

### 5.2 章节级动作
- **Abstract/Intro**：主打 C 线任务定义 + S1 显著性 + S2 协议贡献；**不提**线 A mIoU 数字。
- **Related Work**：按 2.3 表逐篇划界；把负结果链组织成"融合 vs 路由"的张力。
- **Method**（短）：路由原则 → 双场构建（都是标准件）→ α 平滑 + 启用判据。明说"无新架构"。
- **Experiments**（长，论文主体）：
  - 主表 = 2.2 的矩阵（majority 为主口径）
  - 协议节 = query 消融（3 协议）+ 口径敏感性（4 口径）+ McNemar 全表
  - 分析节 = 6 路线失败图谱 + AE 压缩不对称 + waldo 边界条件 + per-label 分层
  - 每个表都有效应量/CI/检验，负结果给机制解释
- **Limitation**：W1/W5/W7 主动写，先于审稿人。

### 5.3 标题候选（避开"改进 LangSplat"框架）
1. "Route by Query Modality: Discriminative-Space-Matched Semantic Fields for Open-Vocabulary 3D Understanding"
2. "Not All Queries Live in CLIP Space: Task-Routed Gaussian Semantic Fields and a Multi-Frame Evidence Protocol for Image-Query 3D Retrieval"
3. "When Does a Second Semantic Field Pay Off? A Routing Principle and Benchmark for Image-Query 3D Retrieval"

---

## 6. 投稿目标（主观概率 = 基于证据链完整度的估计，非承诺）

| 目标 | 匹配度 | 当前证据下概率 | 完成 §7 清单后概率 | 下个 deadline（预计） |
|---|---|---|---|---|
| **RA-L**（8页，滚动） | 高（image-query 检索=机器人定位叙事，审稿周期 ~3 个月） | ~55% | **~70-75%** | 随时可投 |
| **BMVC 2027** | 高（分析型/benchmark 型友好） | ~55-60% | **~70-75%** | 2027-05 |
| **3DV 2028** | 高 | ~55% | ~70% | 2027 秋 |
| WACV 2028 | 中高 | ~50% | ~65-70% | 2027-08 |
| IROS 2027 | 中（需更强机器人动机） | ~45% | ~60-65% | 2027-03 |
| **ICCV 2027** | 中（需要 W1/W2 基本解决） | ~25-30% | ~35-45% | 2027-03 |
| **CVPR 2027** | 中低（线 A SOTA 竞争激烈，我们无方法级增益） | ~15-25% | ~25-35% | 2026-11 中旬（约 5 周后） |
| NeurIPS 2027 D&B track | 中（benchmark 论文专道，若协议包做扎实） | ~30% | ~45-55% | 2027-05 |

**建议主路径**：① 主投 **RA-L**（最快拿到 70%+ 概率的可靠落点，滚动审稿，期间继续补实验）；② 若 6-8 周内完成 §7 清单 P0+P1，改投 **ICCV 2027**（3 月）冲一档；③ BMVC 2027 作为 RA-L 被拒后的会议保底。CVPR 2027（11 月）时间过紧且线 A 卷，**不建议**。

**诚实声明**：任何顶会都没有客观 70% 接收率（CVPR ~23%、ICCV ~26%）；上表 70%+ 仅指"在完成行动清单后，基于证据链完整度与期刊/中档会议匹配度的主观估计"。

---

## 7. 行动清单（按性价比排序，目标把概率推过 70%）

| 优先级 | 项 | 工作量 | 提升点 |
|---|---|---|---|
| **P0** | 跑 LangSplat 官方 eval 协议一行数字（apples-to-apples） | ~2h | 消 W4，防"数字不可比"直接拒稿 |
| **P0** | 代码整理 + README + 一键复现脚本（已大部分在 git） | ~1 天 | 可复现性是 70%+ 的门票 |
| **P1** | 3D-OVS 补 2 个场景（teapot/sofa 等，已有 preprocess 流水线） | ~2 天 | 消 W1 大半；n 上 300+ |
| **P1** | 把 α=0.85 结果（EXP-052，跑批中）并入 α 曲线消融表 | 0（等结果） | 消融完整度 |
| **P2** | OpenGaussian 或 QueryGaussian 适配 image-query 协议跑 1-2 场景 | ~3-5 天 | 消 W2，rebuttal 弹药 |
| **P2** | 定量图：2D tile 快测 Δ vs 3D majority Δ 散点（路由判据可视化） | ~2h | S4 从文字变图 |
| **P3** | DL3DV-OVS 试 1 场景（harder benchmark 前瞻） | ~2 天 | discussion 加分 |
| **P3** | 64d 动态共享内存改造（EXP 方向 6） | ~1 周 | 若成，主表再涨；不成则 limitation 实锤 |

---

## 8. 结论

- **不要**把论文写成"LangSplat 改进版"去线 A 拼 mIoU——那是 IEGaS/GaussDet 的主场，我们没有可比数字。
- **要**把论文写成"image-query 3D 检索的任务定义 + 评估协议 + 路由原则"三位一体——线 C 近乎空白，我们的 53 个实验就是这片空地的第一张系统地图。
- 70%+ 概率的现实路径存在：**RA-L（滚动）或 BMVC 2027，前置完成 P0 清单**；ICCV 2027 为进取选项。
