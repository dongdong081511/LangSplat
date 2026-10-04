# 任务路由双特征场：系统设计（EXP-035a/036a 方法章雏形）

> 状态: 概念验证通过 (teatime +8.2pp) | figurines 跨场景验证进行中 | 本文档为整合设计 v1

## 1. 核心主张

开放词汇 3D 场景理解存在两类本质不同的查询模态，单一特征场无法同时最优：

| 查询模态 | 用户输入 | 任务 | 正确的语义空间 | 实验依据 |
|---------|---------|------|--------------|---------|
| **Text-query** | 文本短语 | 语义分割 (mIoU) | CLIP text-aligned 空间 | EXP-023/025~027: DINO 注入 CLIP 空间全败；EXP-024: CLIP 场 ensemble +1.5~2.3pp |
| **Image-query** | 图像 crop / tile | 空间定位 / 检索 (top-1 命中) | 自监督实例判别空间 (DINOv2) | EXP-035a: DINO 场 90.16% vs CLIP 场 81.97% (+8.2pp)；2D 快测 +10.3pp |

**设计原则：不混合，路由**。与 EXP-023 双流 score 融合失败的本质区别——融合在 text-query 下混合两个 score，DINO 分量必然稀释 softmax(10·sims) 尖峰；任务路由按查询模态把查询派发到独立训练的特征场，两个语义空间互不污染。

## 2. 系统架构

```
输入场景图像
   │
   ├── SAM tile 分割 (共享)
   │
   ├── CLIP tile embedding [512d] ── AE(512→8/24d) ── GS 训练 ──→ CLIP 特征场 (3 levels)
   │                                                        ↓
   └── DINOv2 tile embedding [768d] ─ AE(768→32d) ── GS 训练 ─→ DINO 特征场 (3 levels)

查询路由器 (训练无关, 输入类型判别):
   text string ────────────→ CLIP 场: softmax(τ·sims) relevancy → 分割 mask (mIoU/mAcc)
   image crop / tile ──────→ AE 编码 → DINO 场: 逐像素 cos + 滤波 → argmax 定位 (top-1/top-5)
```

- 两场共享 SAM 分割与相机位姿，各自独立 AE + GS 训练（无交叉损失，彻底隔离）
- 查询路由器是零训练的输入类型判别（文本走 text 端编码器，图像走视觉端）
- 每个场的 AE 维度独立选择（CLIP 场场景相关 8~24d；DINO 场 32d 起步，PCA 下界显示高维更优）

## 3. 评测协议（统一口径）

1. **Text-query 分割**: mIoU / mAcc (multi-prompt mean), relev_temp=10, ±7模板 ensemble（EXP-024 协议）
2. **Image-query 定位**: cross-frame top-1/top-5 命中率，query = GT bbox 最大覆盖 tile 经同空间 AE 编码，database = 其他帧 render 特征场逐像素 cos + 30×30 滤波 argmax 落同类 bbox（eval/eval_image_query.py）
3. **对照组**: 单一 CLIP 场双任务（LangSplat 现状）；单一 DINO 场双任务；混合场（score 融合，EXP-023 已证伪，作反例引用）

## 4. 实验证据链（当前）

| # | 证据 | 结果 | 状态 |
|---|------|------|------|
| 1 | 2D tile 跨帧检索 (teatime) | DINO +10.3pp (74.1% vs 63.8%) | ✓ |
| 2 | 3D 特征场 image-query 检索 (teatime) | DINO 32d 90.16% vs CLIP 8d 81.97% (+8.2pp) | ✓ |
| 3 | 3D 场跨场景 (figurines) | — | 进行中 (EXP-036a) |
| 4 | CLIP 场 text-query 分割 (对照基准) | teatime 0.6932 / figurines 0.5905 (SOTA, EXP-024) | ✓ 已有 |
| 5 | DINO 场 text-query 分割 (反例预期: 应差) | 未测（预期远差于 CLIP 场，反向支撑路由必要性） | 待测 |
| 6 | 混合场 score 融合 (反例) | λ 单调劣化 (EXP-023) | ✓ 已有 |

## 5. 论文叙事骨架（草案）

1. **动机**: LangSplat 系用单一 CLIP 场服务两类查询；实证发现 CLIP 空间对 image-query 检索存在系统性短板（自监督判别特征 vs 图文对齐特征的粒度差异），且 DINO 注入 CLIP 场全败（负结果地图 → 34 实验支撑）
2. **方法**: 任务路由双特征场（§2），零训练路由，管线改动最小（LangSplat 之上增量）
3. **实验**: 双场景 × 双任务矩阵（§4），Loc 任务上 DINO 场大幅超越（+8pp 级），text 任务保持 SOTA 不退化
4. **分析**: 2D/3D 脱钩 + CLIP 空间封闭性机理（EXP-025~030 系列）

## 6. 待办与风险

- [ ] EXP-036a figurines 跨场景（跑中）
- [ ] DINO 场 text-query 反例实验（预期差 → 支撑路由）
- [ ] waldo_kitchen/ramen 第三四场景（主表）
- [ ] 64d 动态共享内存改造（检索精度优化，方向确认后做）
- [ ] 风险: DINO 场增益依赖 GT tile 质量的部分（query 端 tile 与 bbox 的覆盖误差）；跨帧检索协议与 LERF 原版 localization 的差异需在论文中明确声明
- [ ] 命名候选: Task-Routed Semantic Fields / Dual-Query Fields / Routing over Dual Fields
