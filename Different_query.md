# 查询模态解耦语义场（Task-Routed Semantic Fields）——详细教学

> 本文档系统讲解本项目核心创新方法：按查询模态解耦 3DGS 语义场——text-query 走 CLIP 场做开放词汇分割，image-query 走 DINO 场做跨视角定位，并用自蒸馏平滑修复 DINO 场的跨视角一致性。
> 证据链：EXP-001~038（详见 hyper_parameter.md）。

---

## 1. 问题定义：3DGS 语言场面临两类本质不同的查询

3D Gaussian Splatting 语言场（LangSplat 范式）在每个高斯点上附加低维语义特征，渲染任意视角得到逐像素语义特征图。但下游其实存在**两类查询模态**：

| 查询类型 | 输入 | 任务 | 典型应用 |
|---------|------|------|---------|
| **text-query** | 文本短语（"teddy bear"） | 开放词汇分割：找出语义匹配的区域 | 语言指令交互、场景编辑 |
| **image-query** | 图像块（某帧某物体 tile） | 跨视角定位：该物体出现在 3D 场哪些视角哪些位置 | AR 锚定、机器人抓取、"看这里"、2D→3D 检索 |

**关键洞察：这两类查询的度量本质不同**——text-query 需要"图文对齐空间"，image-query 需要"图像-图像实例判别空间"。没有任何单一预训练模型同时最优。

---

## 2. 为什么单场不够：CLIP 与 DINOv2 的互补性（机理层）

### 2.1 CLIP 的优势与死穴

- **优势**：对比学习在 4 亿图文对上训练，text embedding 与 image embedding 同空间 → text-query 的唯一有效语义空间
- **死穴 1（空间封闭性）**：LangSplat 的 relevancy 协议是 `softmax(10·(sim_pos, sim_neg))`，任何非 CLIP 分量注入都会稀释这个尖峰分布。本项目九路实验（PCA blend、cross-attn、InfoNCE、深度第三模态、辅助损失、L14 骨干、双流 score fusion、dense tile-mean、per-patch）全部失败，败因同源——EXP-023 双流实验给出最干净的证明：DINO 投影流与 CLIP 流 cos_sim=0.9256（整体相似）但 score 级融合单调劣化（判别性=噪声）
- **死穴 2（图像-图像判别弱）**：CLIP 的对比目标是图文对齐，不是实例判别。tile 级特征间方差小（EXP-025：pos_sim std 0.037），image-query 检索时"这个物体的特征"和"那个物体的特征"拉不开差距

### 2.2 DINOv2 的优势与死穴

- **优势（实例判别强）**：自监督 DINO 目标就是"不同图块互为负样本"，patch 特征天然判别性强。2D 快测：image-query tile 跨帧检索 DINO 比 CLIP 高 **+5.6pp（figurines）~+10.3pp（teatime）**
- **死穴 1（无文本对齐）**：DINO 特征不在任何文本空间里，无法直接 text-query（LEG-SLAM 尝试 DINO 场做 text-query，需 SAM mask 辅助且性能有限）
- **死穴 2（跨视角一致性低）**：CLIP tile 编码有黑底归一化（mask 外置黑+pad），天然视角不变；DINO 直接看内容，视角敏感——本项目实测同物体跨视角特征相似度 CLIP 0.775/0.683 vs DINO 0.706/0.629（**低 5~7pp**）

### 2.3 被误判的历史信号

EXP-017 早就给出线索：DINOv2 融合的 **IoU 增益跨场景符号翻转（不泛化），但 Loc 增益两场景一致（+3.4/+1.8pp）**。当时的解读是"融合价值在定位"——现在回看，这正是"DINO 的价值在 image-query 任务"的第一次显影，只是当时特征还被硬塞在 CLIP 空间里戴着镣铐。

---

## 3. 方法设计：双独立场 + 零训练路由

### 3.1 总体架构

```
                    ┌─ SAM tile 分割（共享，一次预处理）
输入多视角图像 ──┤
                    └─ 3DGS RGB 几何（共享一次训练，热启动）

场 A（text-query）：CLIP tile global embedding → AE 512→8d/24d → GS 特征场 A
场 B（image-query）：DINOv2 tile 特征 → AE 768→32d → GS 特征场 B（独立训练）

查询路由器（零训练，按输入类型分发）：
  text "chair" ────→ 场 A：softmax(10·relevancy) → 分割 mask
  image patch ─────→ 场 B：特征最近邻检索 → 跨视角定位
```

设计要点：
1. **两场完全独立**——DINO 特征绝不进入 CLIP 空间（绕开封闭性），各自用各自的 AE 和维度最优
2. **几何共享**——两个特征场挂在同一套高斯几何上（热启动 RGB 训好的 ckpt），只训练特征通道；SAM tile 一次提取两场共用
3. **路由零训练**——query 是文本还是图像在系统接口层天然可知，无需学习路由器（这是与"score 融合需要调 λ"的本质区别：没有融合，就没有权重）

### 3.2 为什么"路由"而 不是"融合"（与 EXP-023 的对照）

| | 双流 score fusion（EXP-023，失败） | 任务路由（本方法，成立） |
|---|---|---|
| 融合发生位置 | text-query 协议内混两个 score | 协议外，按任务分发 |
| 权重 | 需要调 λ，且任何混合都稀释 softmax 尖峰 | 无权重 |
| DINO 特征状态 | 被投影进 CLIP 空间（失真） | 原生空间（无损） |
| 失败/成功模式 | λ 单调劣化 | 两场景 image-query 一致占优 |

---

## 4. 自蒸馏平滑（Self-Distillation Smoothing）——DINO 场的跨视角一致性修复

### 4.1 问题

DINO 场直接训练后，多视角平均（GS alpha-blending 天然行为）会稀释判别性：figurines 上 2D 快测 DINO +5.6pp 领先，3D 场检索却 **-11.1pp 落后**（77.78% vs CLIP 24d 场 88.89%）。三因素归因（EXP-036a 诊断）：
1. 入口优势缩水（密集小物体 tile 背景占比高）
2. **放大器效率差异**：3D 多视角融合对 CLIP 放大 +18~30pp、对 DINO 只有 +13~16pp——根源是 DINO 跨视角一致性低，平均时把不一致的部分当噪声抹掉
3. 对照场强度（figurines 的 CLIP 对照是 SOTA 24d 场）

### 4.2 方法

```
第一轮：用原始 DINO 2D tile 特征 T 训练 → 场 G
第二步：用 G 渲染所有视角 → 每帧每 tile 得到渲染特征 R（聚合了跨视角信息的版本）
第三步：构造平滑监督 T' = (1-α)·T + α·R     ← 核心一步，一行凸组合
第四步：用 T' 重新训练场 → 平滑场
```

- **原理**：渲染特征 R 是"3D 一致化后的 DINO 特征"（多视角平均的结果）。把它混回监督信号，等于告诉网络"别的视角看到的这个 tile 长这样"——强制视角不变性，同时保留 (1-α) 份原始判别性
- **α 是"一致性 vs 判别性"旋钮**：
  - figurines 曲线：原始 77.78% → α=0.5 追平 88.89% → **α=0.7 反超 90.74%** → 迭代二次平滑 85.19%（过度稀释）
  - teatime 曲线：原始 90.16% → α=0.7 → 91.80%
  - **单轮 α=0.7 是甜点；迭代无累积效应（二次稀释），不要多轮**
- 类比：这就是知识蒸馏的 self 版本——把模型自己的"集成输出"（渲染=多视角集成）当作教师信号

### 4.3 最终证据矩阵（方法章核心表）

| 场景 | DINO 场（α=0.7） | CLIP 场（各自最优维） | Δ |
|------|-----------------|----------------------|---|
| teatime | **91.80%** | 81.97% (8d) | **+9.8pp** |
| figurines | **90.74%** | 88.89% (24d) | **+1.85pp** |

（image-query 3D 检索 top1，chosen-level；协议见 eval/eval_image_query.py）

---

## 5. 系统级叠加：为什么查询协议增益与路由正交

路由的架构红利：**两个任务各自独立调优，增益直接相加，无干扰**。

- text-query 分割：CLIP 场 + 7 模板 prompt ensemble + 温度 10 → mIoU 0.6932（teatime SOTA）
- image-query 定位：DINO 场 α=0.7 → +9.8pp/+1.85pp
- 若当初走 score 融合路线，这两类增益会在 softmax 里互相稀释；路由设计让它们天然共存

---

## 6. 与相关工作划界（Related Work 写作要点）

| 工作 | 它做什么 | 与本方法的关键差异 |
|------|---------|------------------|
| LangSplat (CVPR24) | CLIP 单场，text-query + Loc | Loc 也是 CLIP 特征自检索（同模态）；无跨模态场 |
| IIN Navigation (2506.07338) | CLIP relevancy 场 + DINOv2 编码目标图做导航 | DINO **没有 3D 场**（只编码 query 图像）；任务为导航；无一致性修复 |
| LEG-SLAM (2506.03073) | DINOv2+PCA 场做 text-query | 替换思路无路由；恰是本方法必要性的反面证据（纯 DINO 场 text-query 性能有限） |
| SemanticSplat / Semantic Gaussians | 多特征融合进单场 | 融合≠路由，text query 为主 |
| OpenGaussian | 点级 CLIP 实例理解 | 支持点击查询但无双场、无 DINO 场 |

**独有贡献组合**：① 双独立场+零训练路由器；② image-query 3D 检索协议+跨模态场对比（首次）；③ 自蒸馏平滑修复跨视角一致性（新技术点，α 单轮+机理归因）；④ CLIP 空间封闭性的九路负结果链作为设计动机（分析贡献）。

---

## 7. 工程复现要点（本项目路径与脚本）

### 7.1 流水线（每场景）

```
1. preprocess --use_dino                 → language_features_*_dino/（DINO tile 特征，768d）
2. 整理 _f_dino.npy/_s_dino.npy          → 从 _s.npy 反推 tile mask（免重跑 SAM）
3. AE 768→32d（默认 lr 1e-4, 100 epochs, epoch>95 才存 ckpt）
4. patch config.h NUM_CHANNELS=32 → 重编译 rasterizer（静态共享内存硬上限 32d；
   64d 需动态共享内存改造）
5. GS 训练×3 level：-m 传基名（train.py 自动加 _L 后缀），热启动 RGB ckpt
6. smooth_tiles.py --alpha 0.7 --iter_tag a07   → 平滑监督集（源=原始 DINO 特征，
   render_base=场景 DINO 场渲染，注意 level 后缀必须正确——后缀丢失会静默退化为复制源特征）
7. 用平滑特征重训场 → render → eval_image_query.py 双场对比
```

### 7.2 关键脚本

| 脚本 | 用途 |
|------|------|
| eval/eval_image_query.py | image-query 3D 检索评测协议（query tile vs 渲染场，chosen-level top1） |
| eval/smooth_tiles.py | 自蒸馏平滑特征生成（--alpha/--src_subdir/--render_base 可调） |
| eval/dino_retrieval_test.py | 2D tile 快测（免训练判别性验证） |
| eval/cross_view_consistency.py | 跨视角同物体一致性诊断（归因工具） |
| eval/render_fidelity_test.py | 渲染忠实度诊断（排除 AE/渲染质量干扰） |

### 7.3 踩坑清单（血泪教训）

1. AE 训练 lr 必须默认 1e-4（1e-3 直接发散）；`--num_epochs` 传入值必须 >95 否则 ckpt 永不落盘
2. train.py 的 `-m` 只传基名，自动追加 `_<level>` 后缀——手工拼全名会导致 ckpt 落错目录（本项目踩了 3 次）
3. smooth_tiles 的 render_base 必须含 level 后缀，且**生成特征后必须做"与源特征差值>阈值"校验**（结果与历史逐分相同=数据流失效的数值指纹）
4. rasterizer 静态共享内存约束：tile 16×16 × dim × 4B < 48KB → **dim ≤ 32**；更高维需改动态共享内存
5. SAM 预处理多场景并行会互卡（0it），必须串行
6. 评测 CLIP 对照时，`--clip_pretrained` 必须与特征教师同源（text/visual 空间错配会出 0.005 的假崩溃）

---

## 8. 未来工作

1. **第三四场景主表**（waldo_kitchen/ramen）：验证路由增益泛化 + 论文 Table 1
2. **DINO 场 64d/128d**：动态共享内存 kernel 改造后重训（PCA 下界显示 128d 还有 ~2.6pp 空间）
3. **query-side ensemble**：image-query 侧多尺度/多 crop tile 特征平均（免训练，可快测）
4. **α 自适应**：按场景物体密度自动选 α（当前 0.7 是两场景甜点，但 teatime 曲线未饱和）
5. **联合训练**：两场共享一个带双头的 AE/特征场，探索参数共享收益（当前完全独立是保守设计）
6. **评估协议升级**：LERF-OVS GT 帧少（teatime 6/figurines 4），构造更大的 image-query 检索 benchmark 本身也是贡献点

---

## 9. 一句话总结

> **Text 查询和图像查询是两种不同的"语言"——前者说 CLIP 的对齐语言，后者说 DINOv2 的判别语言。与其强迫一个场同时讲两种语言（融合，已被九路实验证伪），不如建两个场、在门口指路（路由，零训练），再教 DINO 场把话说稳（自蒸馏平滑，α=0.7 单轮）。**
