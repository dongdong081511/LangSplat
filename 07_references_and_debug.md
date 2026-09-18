# MM-LangSplat 改造方案（分册 07：论文清单、调试与验收）

> **本册定位**：工具书。按需查阅。包含 20+ 篇论文链接、常见故障排除、验收清单。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

---

## 第十阶段：核心论文与引用清单（汇总）

### 10.1 基础框架论文

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| LangSplat: 3D Language Gaussian Splatting (CVPR 2024) | Qin et al. | https://arxiv.org/abs/2312.16084 | https://github.com/minghanqin/LangSplat |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023) | Kerbl et al. | https://repo.sam.inria.fr/fungraph/3d-gaussian-splatting/ | https://github.com/graphdeco-inria/gaussian-splatting |
| Segment Anything (ICCV 2023) | Kirillov et al. | https://arxiv.org/abs/2304.02643 | https://github.com/facebookresearch/segment-anything |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP, ICML 2021) | Radford et al. | https://arxiv.org/abs/2103.00020 | https://github.com/openai/CLIP |
| LERF: Language Embedded Radiance Fields (ICCV 2023) | Kerr et al. | https://lerf.io | https://github.com/lerftoy/lerf |

### 10.2 多模态特征提取论文

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| DINOv2: Learning Robust Visual Features without Supervision (TMLR 2024) | Oquab et al. | https://arxiv.org/abs/2304.07193 | https://github.com/facebookresearch/dinov2 |
| Depth Anything V2 (NeurIPS 2024) | Yang et al. | https://arxiv.org/abs/2406.09414 | https://github.com/DepthAnything/Depth-Anything-V2 |
| DSINE: Revisiting Depth-aware Normal Estimation (ECCV 2024) | Bae et al. | https://arxiv.org/abs/2403.18205 | https://github.com/baegwangbin/DSINE |
| Unsupervised Semantic Segmentation by Distilling Feature Correspondences (STEGO, ICLR 2022) | Hamilton et al. | https://arxiv.org/abs/2207.05026 | https://github.com/hamarb172/STEGO |

### 10.3 多模态融合参考论文（创新点对标）

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields (CVPR 2024) | Zhou et al. | https://arxiv.org/abs/2312.03203 | https://github.com/ShijieZhou-UCLA/feature-3dgs |
| 3D Vision-Language Gaussian Splatting (ICLR 2025) | Peng et al. | https://arxiv.org/abs/2410.07577 | https://github.com/3d-vlgs |
| SemanticSplat: Feed-Forward 3D Scene Understanding with Language-Aware Gaussian Fields (arXiv 2025) | Li et al. | https://arxiv.org/abs/2506.09565 | https://semanticsplat.github.io |
| ShelfGaussian: Shelf-Supervised Open-Vocabulary Gaussian-Based 3D Scene Understanding (CVPR 2026) | Zhao et al. | https://cvpr.thecvf.com/virtual/2026/events/Highlights2026 | https://lunarlab-gatech.github.io |
| FMGS: Foundation Model Embedded 3D Gaussian Splatting (ICLR 2024) | Zuo et al. | https://arxiv.org/abs/2401.01970 | https://github.com/xingxingzuo/FMGS |

### 10.4 融合架构参考论文

| 论文 | 引用 | 链接 | GitHub |
|------|------|------|--------|
| Attention Is All You Need (NeurIPS 2017) | Vaswani et al. | https://arxiv.org/abs/1706.03762 | — |
| Perceiver IO: A General Architecture for Structured Inputs & Outputs (ICLR 2022) | Jaegle et al. | https://arxiv.org/abs/2107.14795 | https://github.com/deepmind/deepmind-research/tree/master/perceiver |
| Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (ICLR 2017) | Shazeer et al. | https://arxiv.org/abs/1701.06538 | — |

### 10.5 数据集论文

| 数据集 | 引用 | 链接 | GitHub |
|--------|------|------|--------|
| LERF Dataset (ICCV 2023) | Kerr et al. | https://lerf.io | https://github.com/lerftoy/lerf |
| 3D-OVS Dataset (NeurIPS 2023) | Liu et al. | https://arxiv.org/abs/2312.03203 | https://github.com/Kunhao-Liu/3D-OVS |

---

## 第十一阶段：调试与故障排除

### 11.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OOM during DINOv2 inference | ViT-L 模型大 | 改用 `dinov2_vitb14`（768d）或 `dinov2_vits14`（384d） |
| DSINE 输出法向量全 0 | 输入未归一化 | 用 ImageNet mean/std 归一化：`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` |
| Cross-attention loss 不收敛 | 学习率太高 | 降到 `1e-5`，加 warmup 1000 步 |
| 渲染 latent 全 0 | rasterizer 不支持非 3 通道 | 用 LangSplat 自带的 `diff-gaussian-rasterization-feature` |
| mIoU 不升 | 融合网络未训好 | 增加 epochs 到 200；检查 CLIP 对齐损失 |
| 训练显存超 | 5 模态全开 + 2.5M 高斯 | 改用 fp16；减小 `d_latent` 到 8；分阶段加载 |

### 11.2 调试技巧

```python
# 检查各模态特征是否合理
import matplotlib.pyplot as plt

def visualize_features(feats):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, (k, v) in zip(axes, feats.items()):
        # PCA 降到 3 维可视化
        v_3d = pca(v[0].permute(1,2,0).reshape(-1, v.shape[1]).cpu().numpy(), 3)
        v_3d = v_3d.reshape(v.shape[2], v.shape[3], 3)
        ax.imshow(v_3d)
        ax.set_title(k)
    plt.savefig('feature_visualization.png')

# 检查 gate weights
def visualize_gate(gate_weights):
    # gate_weights: [B, H, W, n_mods]
    fig, axes = plt.subplots(1, n_mods, figsize=(5*n_mods, 5))
    for i, ax in enumerate(axes):
        ax.imshow(gate_weights[0,:,:,i].cpu().numpy(), cmap='hot')
        ax.set_title(f'Modality {i}')
    plt.savefig('gate_weights.png')
```

---

## 第十二阶段：验收与交付

### 12.1 验收清单

- [ ] 阶段 0：LangSplat baseline 复现，LERF mIoU ≈ 51.4%
- [ ] 阶段 1：所有 5 模态提取器可独立加载并输出正确维度
- [ ] 阶段 2：`MultiModalExtractor` 联合提取可运行
- [ ] 阶段 3：单场景多模态预处理完成，文件结构正确
- [ ] 阶段 4：融合网络训练 loss 收敛（< 0.1）
- [ ] 阶段 5-6：语言高斯训练完成，3D 渲染可生成 d_latent 维 latent
- [ ] 阶段 7：推理可生成 relevancy map
- [ ] 阶段 7：评测 LERF mIoU > 60%（超越 LangSplat baseline 51.4%）
- [ ] 阶段 8：消融实验完成（模态消融 + 融合策略对比 + 维度搜索）
- [ ] 阶段 8：3D-OVS mIoU > 95%（超越 LangSplat baseline 93.4%）

### 12.2 最终交付物

```
output/
├── code/
│   ├── mm_langsplat/                    # 多模态融合模块
│   ├── modified_train.py               # 修改版训练入口
│   └── modified_preprocess.py          # 修改版预处理
├── pretrained/
│   ├── fusion_lerf_ramen.pt
│   ├── fusion_lerf_figurines.pt
│   ├── ... (其他场景)
│   └── mm_langsplat_lerf_ramen.ply
├── results/
│   ├── lerf_results.json               # LERF 评测结果
│   ├── 3dovs_results.json              # 3D-OVS 评测结果
│   ├── ablation_modality.csv           # 模态消融
│   ├── ablation_fusion.csv             # 融合策略对比
│   └── ablation_dim.csv                # 维度搜索
└── figures/
    ├── pareto_curve.png                # mIoU vs 显存 帕累托曲线
    ├── gate_weights_visualization.png   # 模态权重可视化
    └── qualitative_comparison.png      # 与 SOTA 对比可视化
```

---

## 附录 A：核心改动文件清单

| 原文件 | 新文件 | 改动类型 |
|--------|--------|---------|
| `preprocess.py` | `mm_langsplat/preprocess_mm.py` | 新增：5 模态特征提取 |
| `train.py` | `mm_langsplat/trainers/train_langsplat_mm.py` | 新增：训练多模态语言高斯 |
| `gaussian_renderer/__init__.py` | `mm_langsplat/render_language.py` | 新增：渲染语言 latent |
| `scene/gaussian_model.py` | `mm_langsplat/models/mm_language_gaussian.py` | 继承：加 language_latents |
| `eval_langsplat.py` | `mm_langsplat/eval_mm_langsplat.py` | 新增：评测多模态版本 |
| — | `mm_langsplat/extractors/*.py` | 新增：5 个模态提取器 |
| — | `mm_langsplat/fusion/*.py` | 新增：4 种融合策略 |
| — | `mm_langsplat/data/multimodal_dataset.py` | 新增：数据加载器 |
| — | `scripts/run_full_pipeline.sh` | 新增：一键执行脚本 |


---


---
*AI生成*
