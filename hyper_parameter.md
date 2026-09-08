
## 方案B: SAM-CLIP 间 mask 多 crop 多分辨率 (上下文扩展)

> 分支: experiment/multres-B
> 原理: 对每个 SAM mask 在不同上下文尺度裁剪 → 分别送 CLIP → 均值聚合
> 场景: teatime, thresh=0.40

### 网格参数设计

| 编号 | context_scales | 说明 |
|---|---|---|
| B1 | 1.0,1.5 | 2 crops, 中等上下文 |
| B2 | 1.0,2.0 | 2 crops, 宽上下文 |
| B3 | 1.0,1.5,2.0 | 3 crops, 全范围 |
| B4 | 1.0,1.25 | 2 crops, 轻微上下文 |
| B5 | 1.0,1.25,1.5 | 3 crops, 渐进上下文 |

### 设计思路
- B1 vs baseline: 多 crop 是否提升 CLIP 特征质量
- B2 vs B1: 更宽上下文 (2.0x vs 1.5x) 是否更好
- B3: 3 个尺度是否比 2 个更丰富
- B4 vs B1: 轻微上下文 (1.25x) 是否比中等 (1.5x) 更好
- B5: 渐进 3 尺度 vs B1/B3

### 实验结果

| 编号 | context_scales | IoU chosen | Localization acc | vs baseline IoU |
|---|---|---|---|---|
| B1 | 1.0,1.5 | 0.5840 | 0.8644 |
| B2 | 1.0,2.0 | 0.5673 | 0.8644 |
| B3 | 1.0,1.5,2.0 | 0.5071 | 0.7797 |
| B4 | 1.0,1.25 | 0.5697 | 0.7966 |
| B5 | 1.0,1.25,1.5 | 0.5663 | 0.7797 |

## 方案A 泛化性验证 (figurines + waldo_kitchen)

> A1配置 (2级金字塔, cross_iou_thr=0.7) 在其他场景验证泛化性
> baseline对比: 各场景已有产物, 先跑baseline eval获取基线指标

### 实验设计

| 编号 | 场景 | 配置 | thresh | 说明 |
|---|---|---|---|---|
| G1-f | figurines | baseline | 0.45 | figurines基线 |
| G1-a | figurines | A1 (2级,0.7) | 0.45 | figurines方案A |
| G2-f | waldo_kitchen | baseline | 0.40 | waldo_kitchen基线 |
| G2-a | waldo_kitchen | A1 (2级,0.7) | 0.40 | waldo_kitchen方案A |

### 实验结果

| 编号 | 场景 | 配置 | IoU chosen | Localization acc | vs baseline IoU |
|---|---|---|---|---|---|
| G1-f | figurines | baseline | 0.4994 | 0.8036 | — |
| G1-a | figurines | A1 | 0.4730 | 0.8036 | -5.3% |
| G2-f | waldo_kitchen | baseline | 0.4633 | 0.7727 | — |
| G2-a | waldo_kitchen | A1 | 0.4647 | 0.7273 | +0.3% |

## 方案A-Fig: figurines 上采样金字塔 (小物体适配)

> 分支: experiment/multres-A
> 背景: figurines (小雕像) 场景下 A1(下采样0.5x) 导致 IoU -5.3%, 小物体在低分辨率下更模糊
> 思路: 对小物体场景用 pyrUp (2.0x上采样) 而非 pyrDown, 让SAM在高分辨率上分割小物体
> 场景: figurines, thresh=0.45

### 实验设计

| 编号 | pyramid_type | pyramid_levels | cross_iou_thr | 说明 |
|---|---|---|---|---|
| F1 | up | 2 | 0.7 | 上采样2级 (1.0x+2.0x), 默认NMS |
| F2 | up | 2 | 0.5 | 上采样2级, 激进合并 |
| F3 | up | 2 | 0.8 | 上采样2级, 保守合并 |
| F4 | up | 3 | 0.7 | 上采样3级 (1.0x+2.0x+4.0x) |
| F5 | bidir | 3 | 0.7 | 双向 (0.5x+1.0x+2.0x) |

### 设计思路
- F1 vs G1-a: 上采样 vs 下采样, 验证小物体场景方向性
- F2/F3 vs F1: 调整NMS阈值
- F4: 3级上采样, 看更高分辨率是否更好
- F5: 双向金字塔(下采样+上采样), 综合大小尺度

### 实验结果

| 编号 | pyramid_type | pyramid_levels | cross_iou_thr | IoU chosen | Localization acc | vs baseline IoU |
|---|---|---|---|---|---|---|
