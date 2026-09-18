# MM-LangSplat 改造方案（分册 06：推理、评测与消融实验）

> **本册定位**：验证任务。跑推理出 relevancy map，计算 mIoU，产出消融对比表与帕累托曲线。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

---

## 第七阶段：推理与评测

### 7.1 推理流程

**文件**：`mm_langsplat/inference.py`

```python
"""
推理流程：
    1. 加载训练好的 MM-LangSplat 模型
    2. 加载融合网络 + 解码器
    3. 对查询文本：CLIP text encoder -> phi_qry
    4. 渲染 3D 高斯 latent (3 层级) -> F_l
    5. 解码器：F_l -> 512 维 CLIP 嵌入
    6. 计算关联度评分 -> 选最优层级
    7. 输出定位/分割结果
"""

import torch
import open_clip
from mm_langsplat.models.mm_language_gaussian import MMLanguageGaussianModel
from mm_langsplat.fusion.cross_attention import FusionDecoder


def query_3d(model_path, fusion_net_path, text_query, scene_data):
    """开放词汇 3D 查询。"""
    device = torch.device('cuda')
    
    # 1. 加载模型
    gaussians = MMLanguageGaussianModel(d_latent=16)
    gaussians.load_ply(model_path)
    
    ckpt = torch.load(fusion_net_path)
    decoder = FusionDecoder(d_latent=16)
    decoder.load_state_dict(ckpt['decoder'])
    decoder.eval()
    
    # 2. CLIP 文本编码
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    text_token = tokenizer(text_query).to(device)
    with torch.no_grad():
        phi_qry = clip_model.encode_text(text_token)  # [1, 512]
    
    # 3. 渲染三层级 latent
    relevancy_maps = []
    for level in range(3):  # subpart, part, whole
        rendered_latent = gaussians.rasterize_language(scene_data['camera'], level)
        # rendered_latent: [d_latent, H, W]
        
        # 4. 解码到 512 维 CLIP
        with torch.no_grad():
            rendered_clip = decoder(rendered_latent.unsqueeze(0))
        # rendered_clip: [1, 512, H, W]
        
        # 5. 关联度评分（沿用 LangSplat 公式 7）
        relevancy = compute_relevancy(rendered_clip, phi_qry)
        relevancy_maps.append(relevancy)
    
    # 6. 选最优层级
    best_level = max(range(3), key=lambda i: relevancy_maps[i].max())
    
    return relevancy_maps[best_level], best_level


def compute_relevancy(rendered_clip, phi_qry, canon_phrases=['object', 'things', 'stuff', 'texture']):
    """计算关联度评分，沿用 LangSplat 公式 (7)。"""
    # ... 复用 LangSplat 的 relevancy 计算逻辑
    pass
```

### 7.2 评测脚本

**文件**：`mm_langsplat/eval_mm_langsplat.py`

```python
"""
评测脚本。
指标：
    - LERF: Localization Accuracy (%), mIoU (%)
    - 3D-OVS: mIoU (%), Accuracy (%)
对比 SOTA：LangSplat, Occam's LGS, ILGS
"""

import os
import json
import torch
from mm_langsplat.inference import query_3d


def evaluate_lerf(model_path, fusion_net_path, lerf_dir):
    """评测 LERF 数据集。"""
    results = {}
    
    for scene in ['ramen', 'figurines', 'teatime', 'waldo_kitchen']:
        scene_dir = os.path.join(lerf_dir, scene)
        
        # 加载评测查询
        queries = load_lerf_queries(scene_dir)
        
        loc_correct = 0
        seg_ious = []
        
        for q in queries:
            text = q['text']
            gt_loc = q['location']
            gt_mask = q['mask']
            
            # 推理
            relevancy_map, level = query_3d(
                model_path, fusion_net_path, text, scene_dir
            )
            
            # 定位准确率
            pred_loc = relevancy_map.argmax()
            if distance(pred_loc, gt_loc) < threshold:
                loc_correct += 1
            
            # 分割 IoU
            pred_mask = (relevancy_map > 0.5).float()
            iou = (pred_mask * gt_mask).sum() / (pred_mask + gt_mask - pred_mask * gt_mask).sum()
            seg_ious.append(iou)
        
        results[scene] = {
            'loc_accuracy': loc_correct / len(queries),
            'mIoU': sum(seg_ious) / len(seg_ious),
        }
    
    # 汇总
    overall_loc = sum(r['loc_accuracy'] for r in results.values()) / 4
    overall_miou = sum(r['mIoU'] for r in results.values()) / 4
    results['overall'] = {
        'loc_accuracy': overall_loc,
        'mIoU': overall_miou,
    }
    
    return results


if __name__ == '__main__':
    results = evaluate_lerf(
        model_path='pretrained/mm_langsplat_lerf_ramen.ply',
        fusion_net_path='pretrained/fusion_net.pt',
        lerf_dir='data/LERF',
    )
    print(json.dumps(results, indent=2))
```

### 7.3 评测指标

| 数据集 | 任务 | 指标 | LangSplat baseline | MM-LangSplat 目标 |
|--------|------|------|-------------------|-------------------|
| LERF | 3D 物体定位 | Accuracy (%) | 84.3 | **>88** |
| LERF | 3D 语义分割 | mIoU (%) | 51.4 | **>60** |
| 3D-OVS | 3D 语义分割 | mIoU (%) | 93.4 | **>95** |
| 3D-OVS | 3D 语义分割 | Accuracy (%) | 98.9 | **>99** |

---

## 第八阶段：消融实验与维度搜索

### 8.1 模态消融（实验 1）

逐个去掉模态，测 LERF mIoU：

| 配置 | CLIP | DINOv2 | Depth | Normal | Texture | 预期 mIoU |
|------|------|--------|-------|--------|---------|-----------|
| baseline | ✓ | ✗ | ✗ | ✗ | ✗ | 51.4 |
| + DINOv2 | ✓ | ✓ | ✗ | ✗ | ✗ | ~55 |
| + Depth | ✓ | ✗ | ✓ | ✗ | ✗ | ~54 |
| + DINOv2 + Depth | ✓ | ✓ | ✓ | ✗ | ✗ | ~58 |
| + Normal | ✓ | ✓ | ✓ | ✓ | ✗ | ~60 |
| + Texture (full) | ✓ | ✓ | ✓ | ✓ | ✓ | ~65 |

**命令**：
```bash
# 各配置跑一遍
for config in "use_clip only" "use_clip+dino" "use_clip+depth" "use_clip+dino+depth" "use_clip+dino+depth+normal" "all"; do
    python -m mm_langsplat.trainers.train_langsplat_mm \
        --config configs/$config.yml \
        --scene_dir data/LERF/ramen
    python -m mm_langsplat.eval_mm_langsplat --config configs/$config.yml
done
```

### 8.2 融合策略消融（实验 2）

对比 4 种融合方法：

| 融合方法 | LERF mIoU（预期） | 推理速度 | 参数量 |
|---------|------------------|---------|--------|
| Concat + MLP | ~58 | 最快 | 最少 |
| Transformer Encoder | ~62 | 中 | 中 |
| **Cross-Attention（主推）** | **~65** | 中 | 中 |
| MoE | ~63 | 中 | 较多 |

### 8.3 维度 $d$ 搜索（实验 3）

网格搜索 $d \in \{4, 8, 16, 32, 64\}$：

| $d$ | LERF mIoU（预期） | 显存 | 速度 |
|-----|------------------|------|------|
| 4 | ~60 | 30 MB | 最快 |
| 8 | ~63 | 60 MB | 快 |
| **16（推荐）** | **~65** | **120 MB** | **中** |
| 32 | ~66 | 240 MB | 中-慢 |
| 64 | ~65（过拟合） | 480 MB | 慢 |

```bash
# 维度搜索
for d in 4 8 16 32 64; do
    python -m mm_langsplat.trainers.train_fusion \
        --scene_dir data/LERF/ramen \
        --output pretrained/fusion_d${d}.pt \
        --d_latent $d
    python -m mm_langsplat.trainers.train_langsplat_mm \
        --fusion_net_path pretrained/fusion_d${d}.pt \
        --d_latent $d \
        --model_path output/mm_langsplat_d${d}
    python -m mm_langsplat.eval_mm_langsplat \
        --model_path output/mm_langsplat_d${d}/mm_langsplat.ply \
        --d_latent $d
done
```

---


---


---
*AI生成*
