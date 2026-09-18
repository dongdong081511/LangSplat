# MM-LangSplat 改造方案（分册 01：环境搭建与基准复现）

> **本册定位**：项目启动阶段一次性阅读。建立环境、跑通 LangSplat baseline，并安装 5 模态依赖。
> **来源**：从 `LangSplat_cross_attention.md` 切分而来
> **配套**：与其他 6 个分册配合使用，按顺序执行
> **执行日期**：2026-09-14

---

# LangSplat 多模态 Cross-Attention 改造方案（Agent 执行版）

> **文档目的**：作为 agent 的执行手册，将 LangSplat 单模态框架改造为 5 模态 cross-attention 融合的 MM-LangSplat
> **执行方式**：按 8 个阶段顺序执行，每阶段都有明确的输入/输出/验收标准
> **基准代码库**：https://github.com/minghanqin/LangSplat
> **执行日期**：2026-09-07

---

## 第零阶段：环境与基准复现（先跑通原版）

### 0.1 基础环境

```bash
# 克隆 LangSplat 官方仓库
git clone https://github.com/minghanqin/LangSplat.git
cd LangSplat
git submodule update --init --recursive

# 创建 conda 环境
conda create -n mm_langsplat python=3.8 -y
conda activate mm_langsplat

# 安装 PyTorch（与原版一致）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 安装 3D-GS 子模块依赖
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

# 安装 LangSplat 原版 Python 依赖
pip install -r requirements.txt
```

### 0.2 数据集准备

下载 LERF 与 3D-OVS 数据集：
- LERF Dataset: https://github.com/lerftoy/lerf
- 3D-OVS Dataset: https://github.com/Kunhao-Liu/3D-OVS

```bash
# 数据集组织结构
data/
├── LERF/
│   ├── ramen/
│   ├── figurines/
│   ├── teatime/
│   └── waldo_kitchen/
└── 3D-OVS/
    ├── bed/
    ├── bench/
    ├── room/
    ├── sofa/
    └── lawn/
```

### 0.3 基准复现（验收标准：mIoU 51.4%）

按 LangSplat README 跑通：
1. 训练 RGB 场景：`python train.py --config arguments/lego/lerf_ramen.py`
2. 提取 CLIP 特征：`python preprocess.py --source_path data/LERF/ramen/...`
3. 训练语言高斯：`python train.py --config arguments/lego/lerf_ramen_langsplat.py`
4. 评测：`python eval_langsplat.py`

**验收**：在 LERF 上 mIoU ≈ 51.4%，作为后续改进的 baseline。

---

## 第一阶段：环境扩展与多模态依赖安装

### 1.1 各模态模型信息汇总表（agent 速查）

| 模态 | 模型 | 引用论文 | 论文链接 | GitHub 仓库 | 安装方式 |
|------|------|---------|---------|------------|---------|
| CLIP（语义） | OpenCLIP ViT-B/16 | Radford et al., ICML 2021, "Learning Transferable Visual Models From Natural Language Supervision" | https://arxiv.org/abs/2103.00020 | https://github.com/openai/CLIP | `pip install open_clip_torch` |
| DINOv2（视觉） | DINOv2 ViT-L/14 | Oquab et al., TMLR 2024, "DINOv2: Learning Robust Visual Features without Supervision" | https://arxiv.org/abs/2304.07193 | https://github.com/facebookresearch/dinov2 | `pip install -e .` 或 torch.hub |
| Depth（深度） | Depth Anything V2（ViT-L） | Yang et al., NeurIPS 2024, "Depth Anything V2" | https://arxiv.org/abs/2406.09414 | https://github.com/DepthAnything/Depth-Anything-V2 | 下载权重，直接 import |
| Normal（法向量） | DSINE | Bae et al., ECCV 2024, "DSINE: Revisiting Depth-aware Normal Estimation" | https://arxiv.org/abs/2403.18205 | https://github.com/baegwangbin/DSINE | clone + 加载权重 |
| Texture（纹理） | STEGO | Hamilton et al., ICLR 2022, "Unsupervised Semantic Segmentation by Distilling Feature Correspondences" | https://arxiv.org/abs/2207.05026 | https://github.com/hamarb172/STEGO | clone + install |
| SAM（分割） | SAM ViT-H | Kirillov et al., ICCV 2023, "Segment Anything" | https://arxiv.org/abs/2304.02643 | https://github.com/facebookresearch/segment-anything | `pip install git+https://github.com/facebookresearch/segment-anything.git` |

### 1.2 安装多模态依赖

```bash
# OpenCLIP（CLIP 实现，与 LangSplat 一致）
pip install open_clip_torch==2.24.0

# DINOv2（通过 torch.hub 加载，无需 clone）
# 已包含在 torch.hub 中，运行时自动下载

# Depth Anything V2
cd ${PROJECT_ROOT}/third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt
# 下载预训练权重
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
cd ${PROJECT_ROOT}

# DSINE（法向量）
cd ${PROJECT_ROOT}/third_party
git clone https://github.com/baegwangbin/DSINE.git
# 下载权重
cd DSINE
wget https://huggingface.co/baegwangbin/DSINE/resolve/main/dsine.pt
cd ${PROJECT_ROOT}

# STEGO（纹理）
cd ${PROJECT_ROOT}/third_party
git clone https://github.com/hamarb172/STEGO.git
cd STEGO
pip install -r requirements.txt
# 下载 ImageNet 预训练权重
mkdir -p logs && cd logs
wget https://huggingface.co/hamarb172/STEGO/resolve/main/redirected_latest.pt
cd ${PROJECT_ROOT}

# SAM（LangSplat 已有，确保安装）
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 1.3 第三方依赖目录结构（推荐）

```
LangSplat/
├── third_party/
│   ├── Depth-Anything-V2/
│   ├── DSINE/
│   └── STEGO/
├── pretrained/
│   ├── depth_anything_v2_vitl.pth
│   ├── dsine.pt
│   ├── redirect_latest.pt
│   ├── sam_vit_h_4b8939.pth
│   └── dinov2_vitl14_pretrain.pth
├── mm_langsplat/                   # 新建：多模态融合模块
│   ├── __init__.py
│   ├── extractors/                # 各模态提取器
│   │   ├── __init__.py
│   │   ├── clip_extractor.py
│   │   ├── dino_extractor.py
│   │   ├── depth_extractor.py
│   │   ├── normal_extractor.py
│   │   ├── texture_extractor.py
│   │   └── sam_extractor.py
│   ├── fusion/                    # 融合网络
│   │   ├── __init__.py
│   │   ├── cross_attention.py     # 主推方案
│   │   ├── concat_mlp.py          # baseline
│   │   ├── transformer_encoder.py # baseline
│   │   └── moe.py                 # baseline
│   ├── models/
│   │   ├── __init__.py
│   │   └── mm_language_gaussian.py  # 多模态语言高斯模型
│   ├── data/
│   │   ├── __init__.py
│   │   └── multimodal_dataset.py
│   └── trainers/
│       ├── __init__.py
│       ├── train_fusion.py        # 训练融合网络
│       └── train_langsplat_mm.py  # 训练语言高斯
└── ... (LangSplat 原有文件)
```

**验收**：
- 所有 `import` 可正常加载
- 所有预训练权重文件存在并 sha256 校验通过
- DINOv2 `torch.hub.load(...)` 可正常加载



---


---
*AI生成*
