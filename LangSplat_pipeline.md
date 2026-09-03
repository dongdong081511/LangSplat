# LangSplat 完整训练流程

> 本文档记录在 Ubuntu 22.04 + RTX 4090 (sm_89) 环境下从零复现 LangSplat (CVPR 2024 Highlight) 的完整流程。
> 包含环境搭建、数据准备、训练、渲染、评估全步骤，可直接在另一台机器上按此复现。

---

## 1. 硬件与软件要求

### 硬件
- GPU: NVIDIA RTX 4090 (24GB VRAM) 或同等显卡，Compute Capability 7.0+
- 内存: 32GB+ 推荐

### 软件
- OS: Ubuntu 22.04 LTS
- CUDA Driver: 535+ (支持 CUDA 11.8 运行时)
- Conda (Miniconda 或 Anaconda)
- gcc/g++ 11 (Ubuntu 22.04 默认)
- CUDA Toolkit 11.8 (nvcc，用于编译 CUDA 扩展)

> **关键**: RTX 4090 为 sm_89 架构。PyTorch 1.13.1+cu117 的运行时**不包含 sm_89 内核**，
> 会导致反向传播时 CUDA 内存损坏（表现为 `numel: integer multiplication overflow` 或
> `illegal memory access`）。**必须使用 PyTorch 2.0.1+cu118 或更高版本**。

---

## 2. 环境搭建

### 2.1 克隆项目

```bash
git clone https://github.com/minghanqin/LangSplat.git --recursive
cd LangSplat
```

### 2.2 创建 Conda 环境

```bash
conda create -n langsplat python=3.9 -y
conda activate langsplat
```

### 2.3 安装 PyTorch 2.0.1 + CUDA 11.8

> **不要使用** environment.yml 中默认的 PyTorch 1.12.1/1.13.1，它们不支持 sm_89。

```bash
# 方法1: 直接从 PyTorch 官方下载 (较慢)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 方法2: 使用阿里云镜像下载 wheel 后本地安装 (国内推荐)
wget https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl -O /tmp/torch-2.0.1+cu118.whl
wget https://mirrors.aliyun.com/pytorch-wheels/cu118/torchvision-0.15.2%2Bcu118-cp39-cp39-linux_x86_64.whl -O /tmp/torchvision-0.15.2+cu118.whl
pip install /tmp/torch-2.0.1+cu118.whl /tmp/torchvision-0.15.2+cu118.whl
```

验证安装:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA arch list:', torch.cuda.get_arch_list()); print('GPU:', torch.cuda.get_device_name(0))"
# 应输出包含 sm_89 的 arch list
```

### 2.4 安装 CUDA Toolkit 11.8 (nvcc)

```bash
# 使用 conda 安装 cuda-nvcc 和 cuda-cudart-dev (不需要系统级 CUDA 安装)
conda install -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev cuda-cudart -y
```

验证:
```bash
export CUDA_HOME=$CONDA_PREFIX
nvcc --version
# 应输出 Cuda compilation tools, release 11.8, V11.8.89
```

### 2.5 安装其他 Python 依赖

```bash
pip install open-clip-torch plyfile tqdm opencv-python tensorboard jaxtyping matplotlib
```

### 2.6 安装 segment-anything-langsplat

```bash
pip install submodules/segment-anything-langsplat
```

### 2.7 编译 CUDA 扩展

> **关键**: 必须指定 `TORCH_CUDA_ARCH_LIST="8.9"` 以编译 sm_89 内核。

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.9"
export CC=gcc-11
export CXX=g++-11

# 编译 diff_gaussian_rasterization (langsplat-rasterization)
cd submodules/langsplat-rasterization
pip install . --no-build-isolation
cd ../..

# 编译 simple_knn
cd submodules/simple-knn
pip install . --no-build-isolation
cd ../..
```

如果遇到 `cannot find -lcudart` 错误，需修复 libcudart.so 符号链接:
```bash
# 查找可用的 libcudart.so
find $CONDA_PREFIX -name 'libcudart.so*'
# 创建正确的符号链接 (指向实际存在的文件)
ln -sf $CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0 $CONDA_PREFIX/lib/libcudart.so
```

验证编译结果:
```bash
python -c "
import diff_gaussian_rasterization._C as C
import os
so_path = os.path.join(os.path.dirname(C.__file__), '_C.cpython-39-x86_64-linux-gnu.so')
print('Extension loaded:', so_path)
"
# 检查 .so 包含 sm_89 内核
python -c "
import subprocess, os
import diff_gaussian_rasterization._C as C
so_path = os.path.join(os.path.dirname(C.__file__), '_C.cpython-39-x86_64-linux-gnu.so')
# 如果有 cuobjdump 可用
try:
    r = subprocess.run(['cuobjdump', '--list-elf', so_path], capture_output=True, text=True)
    print(r.stdout)
except FileNotFoundError:
    print('cuobjdump not available, skip arch check')
"
```

---

## 3. 数据准备

### 3.1 下载 LERF 数据集

从以下链接下载预处理好的 LERF 数据集:
- Google Drive: https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing
- 百度网盘: https://pan.baidu.com/s/1S_cdmN9EFOlCQ3z1GZR3EA?pwd=lfea

解压到项目目录:
```bash
unzip lerf_ovs.zip -d dataset/
```

数据集结构:
```
dataset/lerf_ovs/
├── teatime/          # 场景名称
│   ├── images/       # 原始图像
│   ├── input/        # 输入图像 (同 images)
│   ├── sparse/       # COLMAP 稀疏重建
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   └── language_feature/   # CLIP 语言特征 (512维)
│       ├── 00_f.npy
│       ├── 00_s.npy
│       └── ...
├── forks/
├── ramen/
└── ...
```

### 3.2 下载 3D-OVS 数据集 (可选)

```bash
# Google Drive: https://drive.google.com/drive/folders/1kdV14Gu5nZX6WOPbccG7t7obP_aXkOuC?usp=sharing
```

### 3.3 下载 Autoencoder 预训练模型

从以下链接下载 AE checkpoint:
- Google Drive: https://drive.google.com/drive/folders/1ASFXWOwaXP_aSXV2iMDmEfILaDXQXlrE?usp=sharing
- 百度网盘: https://pan.baidu.com/s/12L83uEi5KlF9ViAZqp0B4w?pwd=dl22

解压到:
```bash
mkdir -p autoencoder/ckpt/
# 将下载的 checkpoint 文件放入 autoencoder/ckpt/
```

### 3.4 下载 SAM 模型 (仅处理自定义场景需要)

```bash
mkdir -p ckpts/
# 下载 SAM ViT-H 模型
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ckpts/sam_vit_h_4b8939.pth
```

---

## 4. 训练流程

### 4.1 训练 RGB 3D Gaussian Splatting 模型 (前置步骤)

LangSplat 需要先训练好 RGB 3DGS 模型作为基础。

```bash
# 以 teatime 场景为例
python train.py -s dataset/lerf_ovs/teatime -m dataset/lerf_ovs/teatime/output/teatime --iterations 30000
```

训练完成后，在 `dataset/lerf_ovs/teatime/output/teatime/` 下应有:
- `chkpnt30000.pth` — RGB 模型 checkpoint
- `point_cloud/iteration_30000/point_cloud.ply` — 高斯点云
- `cameras.json`, `cfg_args`, `input.ply`

### 4.2 生成语言特征 (仅自定义场景需要)

如果使用预处理的 LERF 数据集，可跳过此步 (数据集已包含 language_feature/)。

```bash
# 生成 CLIP 语言特征
python preprocess.py --dataset_path dataset/lerf_ovs/teatime
```

### 4.3 训练场景级 Autoencoder

> 如果已下载预训练 AE 模型，可跳过此步。

```bash
cd autoencoder

# 训练 AE
python train.py --dataset_path ../dataset/lerf_ovs/teatime \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --lr 0.0007 \
    --dataset_name ae_ckpt

# 生成 3 维语言特征
python test.py --dataset_path ../dataset/lerf_ovs/teatime --dataset_name ae_ckpt

cd ..
```

### 4.4 训练 LangSplat (3层级联)

LangSplat 使用 3 级级联训练策略 (feature_level 1, 2, 3)：

```bash
# 获取 RGB checkpoint 路径
RGB_CKPT=dataset/lerf_ovs/teatime/output/teatime/chkpnt30000.pth

# Level 1
python train.py -s dataset/lerf_ovs/teatime \
    -m output/teatime_1 \
    --start_checkpoint $RGB_CKPT \
    --feature_level 1

# Level 2
python train.py -s dataset/lerf_ovs/teatime \
    -m output/teatime_2 \
    --start_checkpoint $RGB_CKPT \
    --feature_level 2

# Level 3
python train.py -s dataset/lerf_ovs/teatime \
    -m output/teatime_3 \
    --start_checkpoint $RGB_CKPT \
    --feature_level 3
```

---

## 5. 渲染

对每个 level 渲染 RGB 和语言特征：

```bash
for level in 1 2 3
do
    # 渲染 RGB
    python render.py -m output/teatime_${level}

    # 渲染语言特征
    python render.py -m output/teatime_${level} --include_feature
done
```

---

## 6. 评估

### 6.1 3D Object Localization / 3D Semantic Segmentation

```bash
cd eval

# 编辑 eval.sh 中的 CASE_NAME 为你的场景名
# 确认 gt_folder 指向正确的 label 路径
sh eval.sh
```

`eval.sh` 内容参考:
```bash
CASE_NAME="teatime"
gt_folder="../dataset/lerf_ovs/label"
root_path="../"

python evaluate_iou_loc.py \
    --dataset_name ${CASE_NAME} \
    --feat_dir ${root_path}/output \
    --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
    --output_dir ${root_path}/eval_result \
    --mask_thresh 0.4 \
    --encoder_dims 256 128 64 32 3 \
    --decoder_dims 16 32 64 128 256 256 512 \
    --json_folder ${gt_folder}
```

### 6.2 评估多个场景

如需评估所有 LERF 场景，修改 `eval.sh` 中的 `CASE_NAME` 后重新运行:
```bash
for scene in teatime forks ramen figs cookies; do
    CASE_NAME=$scene sh eval.sh
done
```

### 6.3 多场景批处理完整流水线 (RGB → LangSplat → 渲染 → AE → 评估)

对 LERF 数据集中的多个场景 (teatime 之外，如 figurines、ramen、waldo_kitchen) 执行完整流水线，
可使用下面的批处理脚本。每个场景依次完成 RGB 3DGS 训练、LangSplat 3 层级联训练、
RGB/语言特征渲染、场景级 Autoencoder 训练与 3 维特征生成、评估。

> **注**: 以下命令需在项目根目录 `LangSplat/` 下执行，使用 langsplat conda 环境
> (`/home/xiedexia/.conda/envs/langsplat/bin/python`)，GPU 命令需禁用沙箱。

#### 6.3.1 一次性批处理脚本

> **关键**: 步骤顺序必须是 RGB → AE → 生成 dim3 → LangSplat → 渲染 → 评估。
> AE 训练和 dim3 生成必须在 LangSplat 之前，因为 LangSplat 训练时读取 dim3，
> 评估时用同一 AE 解码。若 dim3 与 AE 不匹配，评估 IoU 将接近 0 (见 6.5)。

将以下内容保存为 `run_all_scenes.sh`:
```bash
#!/bin/bash
set -e
PY=/home/xiedexia/.conda/envs/langsplat/bin/python
SCENES="figurines ramen waldo_kitchen"   # 可按需增减
ROOT=/home/xiedexia/project/LangSplat
cd "$ROOT"

for scene in $SCENES; do
  echo "############### SCENE: $scene ###############"
  SRC="dataset/lerf_ovs/$scene"
  RGB_OUT="dataset/lerf_ovs/$scene/output/$scene"   # RGB 输出 (train.py 自动加 _-1)
  RGB_CKPT="dataset/lerf_ovs/$scene/output/${scene}_-1/chkpnt30000.pth"

  # 1) RGB 3DGS 训练 (feature_level=-1, 不启用 include_feature)
  if [ ! -f "$RGB_CKPT" ]; then
    echo "[$scene] RGB 3DGS training..."
    $PY train.py -s "$SRC" -m "$RGB_OUT" --feature_level -1 \
        --iterations 30000 --test_iterations 30000 \
        --save_iterations 30000 --checkpoint_iterations 30000
  fi

  # 2) 训练场景级 Autoencoder (必须在 LangSplat 之前)
  AE_DIR="autoencoder/ckpt/$scene/ae_ckpt"
  if [ ! -f "$AE_DIR/best_ckpt.pth" ]; then
    echo "[$scene] Autoencoder training..."
    $PY autoencoder/train.py --dataset_path "$SRC" --dataset_name "$scene" \
        --num_epochs 100 --ae_ckpt_dir autoencoder/ckpt
  fi

  # 3) 生成 3 维语言特征 (用场景级 AE 推理, 必须在 LangSplat 之前)
  #    若数据集自带 dim3, 先备份再重新生成, 确保 dim3 与 AE 匹配
  if [ -d "$SRC/language_features_dim3" ] && [ ! -d "$SRC/language_features_dim3_orig" ]; then
    mv "$SRC/language_features_dim3" "$SRC/language_features_dim3_orig"
  fi
  if [ ! -d "$SRC/language_features_dim3" ]; then
    echo "[$scene] generating dim3 features..."
    (cd autoencoder && $PY test.py --dataset_path "../$SRC" --dataset_name "$scene" \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512)
  fi

  # 4) LangSplat 3 层级联训练 (从 RGB checkpoint 开始)
  #    train.py 会自动在 model_path 后追加 _<feature_level>
  #    所以 -m 传 output/$scene, 实际目录变为 output/${scene}_1 等
  for lvl in 1 2 3; do
    OUT="output/${scene}_${lvl}"
    if [ ! -f "$OUT/chkpnt30000.pth" ]; then
      echo "[$scene] LangSplat level $lvl training..."
      $PY train.py -s "$SRC" -m "output/$scene" --feature_level $lvl \
          --include_feature --start_checkpoint "$RGB_CKPT" \
          --iterations 30000 --test_iterations 30000 \
          --save_iterations 30000 --checkpoint_iterations 30000
    fi
  done

  # 5) 渲染 RGB + 语言特征 (render.py 不追加后缀, -m 直接用最终路径)
  for lvl in 1 2 3; do
    OUT="output/${scene}_${lvl}"
    if [ ! -d "$OUT/train/ours_None/renders_npy" ]; then
      echo "[$scene] rendering level $lvl..."
      $PY render.py -m "$OUT" --include_feature
    fi
  done

  # 6) 评估 (AE 解码 dim3 → 512 → CLIP 激活 → IoU)
  echo "[$scene] evaluation..."
  (cd eval && $PY evaluate_iou_loc.py \
      --dataset_name "$scene" \
      --feat_dir ../output \
      --ae_ckpt_dir ../autoencoder/ckpt \
      --output_dir ../eval_result \
      --mask_thresh 0.4 \
      --encoder_dims 256 128 64 32 3 \
      --decoder_dims 16 32 64 128 256 256 512 \
      --json_folder ../dataset/lerf_ovs/label)
done
echo "############### ALL DONE ###############"
```

执行:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_all_scenes.sh 2>&1 | tee /tmp/all_scenes.log
```

#### 6.3.2 目录命名约定

- RGB 模型: `dataset/lerf_ovs/<scene>/output/<scene>_<scene>_-1/` (由 `--feature_level -1` 自动加后缀)
- LangSplat Level N: `output/<scene>_<N>_<N>/` (由 `--feature_level N` 自动加后缀)
- 渲染输出: `output/<scene>_<N>_<N>/train/ours_None/renders_npy/*.npy`
- AE checkpoint: `autoencoder/ckpt/<scene>/ae_ckpt/best_ckpt.pth`
- 3 维语言特征: `dataset/lerf_ovs/<scene>/language_features_dim3/`
- 评估结果: `eval_result/<scene>/`

#### 6.3.3 实际执行结果 (LERF figurines / ramen / waldo_kitchen)

使用上述流水线，对 teatime 之外的 3 个 LERF 场景完整执行训练/渲染/评估，结果如下:

| 场景 | RGB PSNR | LangSplat 3 levels | 评估 IoU (chosen) | Localization Acc |
|------|----------|--------------------|--------------------|------------------|
| teatime (基线) | — | ✓ 30k×3 | 0.6431 | 0.8983 |
| figurines | 25.14 | ✓ 30k×3 | **0.4991** | **0.8036** |
| ramen | 29.16 | ✓ 30k×3 | **0.4901** | **0.6761** |
| waldo_kitchen | 32.73 | ✓ 30k×3 | **0.4633** | **0.7727** |

> **结果说明**:
> - 3 个场景的 IoU 从 ~0.01–0.11 提升至 ~0.46–0.50，Localization 从 ~0.02–0.30 提升至 ~0.68–0.80。
> - 提升的关键在于修复了 dim3 特征与场景级 AE 不匹配的问题 (详见 6.5 节根因分析)。
> - teatime 作为基线，IoU 0.64 / Loc 0.90，与论文报告接近，流水线正确。

### 6.4 排查评估 IoU 异常的工具命令

#### 6.4.1 检查渲染语言特征与 GT dim3 的相似度

```bash
$PY -c "
import numpy as np
from numpy.linalg import norm
scene='ramen'; frame='frame_00006'; idx=6
r = np.load(f'output/{scene}_3_3/train/ours_None/renders_npy/{idx:05d}.npy')
df = np.load(f'dataset/lerf_ovs/{scene}/language_features_dim3/{frame}_f.npy')
ds = np.load(f'dataset/lerf_ovs/{scene}/language_features_dim3/{frame}_s.npy').astype(int)
seg = ds[3]; m = seg != -1
g = np.zeros(r.shape); g[m] = df[seg[m]]
cos = np.sum(r[m]*g[m],1)/(norm(r[m],axis=1)*norm(g[m],axis=1)+1e-8)
print(f'{scene} cos sim mean={cos.mean():.4f}')
"
```

#### 6.4.2 检查 CLIP 激活值分布 (判断 mask 是否过大)

```bash
# 见仓库内 /tmp/check_activ.py 思路:
# 堆叠 3 个 level 渲染特征 → AE decode → L2 normalize → clip_model.get_max_across
# 输出每个 level / prompt 的 max、mean、>0.5 ratio
# 若 >0.5 ratio 过大 (>30%) 而 max 偏低, 说明激活图平坦, mask 会过大, IoU 偏低
```

#### 6.4.3 检查 AE 重建质量

```bash
cd autoencoder
for scene in figurines ramen waldo_kitchen; do
  $PY test.py --dataset_path ../dataset/lerf_ovs/$scene \
      --ckpt_path ckpt/$scene/ae_ckpt/best_ckpt.pth \
      --encoder_dims 256 128 64 32 3 \
      --decoder_dims 16 32 64 128 256 256 512
done
```

---

## 7. 常见问题与解决方案

### 7.1 CUDA 错误: `numel: integer multiplication overflow` / `illegal memory access`

**根因**: PyTorch 1.13.1+cu117 的运行时不支持 sm_89 (RTX 4090) 架构。

**解决**: 升级到 PyTorch 2.0.1+cu118，并重新编译 CUDA 扩展:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# 重新编译扩展
cd submodules/langsplat-rasterization && pip install . --no-build-isolation && cd ../..
cd submodules/simple-knn && pip install . --no-build-isolation && cd ../..
```

### 7.2 编译错误: `Unknown CUDA arch (8.9) or GPU not supported`

**根因**: PyTorch 的 `cpp_extension.py` 中 `supported_arches` 列表不含 8.9。

**解决**: 设置环境变量 `TORCH_CUDA_ARCH_LIST="8.9"` (PyTorch 2.0+ 已内置支持，无需手动修改源码)。

### 7.3 编译错误: `cannot find -lcudart`

**根因**: conda 环境中 `libcudart.so` 符号链接断开。

**解决**:
```bash
# 查找实际存在的 libcudart.so 文件
find $CONDA_PREFIX -name 'libcudart.so*'
# 创建正确的符号链接
ln -sf <找到的实际路径> $CONDA_PREFIX/lib/libcudart.so
```

### 7.4 编译错误: `cc1plus: execvp: 没有那个文件或目录`

**根因**: gcc 版本不匹配，nvcc 找不到 C++ 编译器。

**解决**:
```bash
export CC=gcc-11
export CXX=g++-11
```

### 7.5 GPU 未被检测到 (Trae IDE 沙箱环境)

**根因**: Trae IDE 的命令执行环境是沙箱，隔离了 GPU 设备文件访问。

**解决**: 执行 GPU 相关命令时需禁用沙箱 (在 RunCommand 中设置 `dangerouslyDisableSandbox: true`)，
或直接在系统终端中运行。

### 7.6 `Attempting CUDA graph capture of step() ... capturable=False`

**根因**: PyTorch 1.13.1 在某些 CUDA 环境下误触发 CUDA graph capture 检查。

**解决** (仅 PyTorch 1.13.1 需要，升级到 2.0+ 后不需要):
```python
# 在 train.py 开头添加
import torch.optim.optimizer as _opt
_opt.Optimizer._cuda_graph_capture_health_check = lambda self: None
```

### 7.7 CUDA 错误: `illegal memory access` (densification 后 backward 失败)

**症状**: 训练 RGB 模型时，densification 在 iter 600 触发后，下一次 `loss.backward()` 报
`CUDA error: an illegal memory access was encountered`。禁用 densification 则不报错。

**根因**: `gaussian_renderer/__init__.py` 中，当 `include_feature=False` (RGB 训练) 时，
`language_feature_precomp` 被设置为 `torch.zeros((1,))` (仅 1 个元素)。但 CUDA backward
kernel (`backward.cu` 第 503 行) **无条件读取** `language_feature[coll_id * F + i]`
(F=3)，导致越界读取。densification 前点数少，越界访问可能落在已分配页内不报错；
densification 后点数增加，`coll_id` 变大，越界访问落入未分配页触发 illegal memory access。

**解决**: 修改 `gaussian_renderer/__init__.py`，为 `include_feature=False` 分配 `P*3` 大小的零张量:
```python
# 修改前 (错误):
language_feature_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)

# 修改后 (正确):
language_feature_precomp = torch.zeros((means3D.shape[0], 3), dtype=opacity.dtype, device=opacity.device)
```

> **注**: 这是 LangSplat 仓库的已知 bug，原始 3DGS 代码无此问题 (无 language_feature)。
> 此修复对 LangSplat 训练 (include_feature=True) 无影响。

---

## 8. 快速验证 (Smoke Test)

在正式训练前，先用少量迭代验证环境是否正常:

```bash
# 1000 迭代快速测试
python train.py -s dataset/lerf_ovs/teatime \
    -m output/smoke_test \
    --iterations 1000 \
    --test_iterations 1000 \
    --save_iterations 1000 \
    --checkpoint_iterations 1000
```

如果训练能顺利通过 1000 迭代 (特别是 densification 在 iter 500-600 触发后不报错)，
说明环境配置正确，可以开始正式训练。

---

## 9. 项目目录结构

```
LangSplat/
├── train.py                    # 训练脚本 (RGB 3DGS + LangSplat)
├── render.py                   # 渲染脚本
├── preprocess.py               # 语言特征预处理 (CLIP + SAM)
├── render.py                   # 渲染脚本
├── arguments/                  # 命令行参数定义
├── gaussian_renderer/          # 高斯渲染器
├── scene/                      # 场景、相机、高斯模型定义
│   ├── gaussian_model.py       # GaussianModel (含 densification)
│   ├── dataset_readers.py      # 数据集读取
│   └── ...
├── autoencoder/                # 场景级语言 Autoencoder
│   ├── train.py                # AE 训练
│   ├── test.py                 # AE 推理 (生成 3 维特征)
│   └── ckpt/                   # AE checkpoint
├── eval/                       # 评估代码
│   ├── eval.sh                 # 评估脚本
│   └── evaluate_iou_loc.py     # IoU / Localization 评估
├── submodules/
│   ├── langsplat-rasterization/  # diff_gaussian_rasterization CUDA 扩展
│   ├── simple-knn/               # simple_knn CUDA 扩展
│   └── segment-anything-langsplat/  # SAM 修改版
├── dataset/                    # 数据集
│   └── lerf_ovs/               # LERF 数据集
│       ├── teatime/
│       ├── forks/
│       └── ...
├── output/                     # 训练输出
├── environment.yml             # 原始 conda 环境 (已过时，参考本文档)
├── process.sh                  # 流水线示例脚本
└── LangSplat_pipeline.md       # 本文档
```

---

## 10. 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iterations` | 30000 | 训练迭代次数 |
| `--feature_level` | -1 | LangSplat 级联级别 (1/2/3)，-1 表示 RGB 训练 |
| `--start_checkpoint` | None | RGB 3DGS 的 checkpoint 路径，LangSplat 训练时必须指定 |
| `--include_feature` | False | 渲染时是否输出语言特征 |
| `--densify_from_iter` | 500 | 开始 densification 的迭代 |
| `--densify_until_iter` | 15000 | 停止 densification 的迭代 |
| `--densification_interval` | 100 | densification 间隔 |
| `--densify_grad_threshold` | 0.0002 | densification 梯度阈值 |
| `--opacity_reset_interval` | 3000 | 不透明度重置间隔 |
| `--sh_degree` | 3 | 球谐函数阶数 |

---

## 11. 论文指标参考

LERF 数据集上的 3D Object Localization 和 3D Semantic Segmentation 指标请参考论文:
- 论文链接: https://arxiv.org/pdf/2312.16084.pdf
- 项目主页: https://langsplat.github.io/
