# LangSplat 项目规则 (Project Rules)

> 本文件是 Trae AI 在本仓库工作时的强制约束规范，优先级高于通用约定。

---

## 1. 项目简介

- **项目名**：LangSplat —— *3D Language Gaussian Splatting*（CVPR 2024 Highlight 论文复现 / 扩展）
- **项目路径**：`/home/xiedexia/project/LangSplat`
- **项目规模**：约 682 GB（含大量模型权重、点云、数据集与中间产物）
- **核心目标**：在 3D 高斯泼溅（3DGS）中嵌入 CLIP 语言特征，实现 3D 场景的开放词汇语义分割与文本查询
- **关键创新**：用场景级 Autoencoder 将 512 维 CLIP 特征降维到 3 维，缓解显式建模的显存压力；采用 3 层级联（feature_level 1/2/3）训练策略
- **论文**：[arXiv 2312.16084](https://arxiv.org/pdf/2312.16084.pdf) ｜ [项目主页](https://langsplat.github.io/)

---

## 2. 技术栈

| 项 | 版本 / 说明 |
|---|---|
| Python | **3.9.25**（位于 `/home/xiedexia/.conda/envs/langsplat/bin/python`）|
| PyTorch | 1.12.1（CUDA 11.6 / cudatoolkit=11.6）|
| torchvision | 0.13.1 |
| CUDA Compute Capability | ≥ 7.0，建议 24 GB VRAM |
| 3DGS | 自定义光栅化扩展 `submodules/langsplat-rasterization` |
| CLIP | OpenCLIP（`ViT-B-16`, pretrained=`laion2b_s34b_b88k`）|
| SAM | `submodules/segment-anything-langsplat`（预处理分割）|
| 关键依赖 | `open-clip-torch`、`plyfile=0.8.1`、`opencv`、`tensorboard`、`jaxtyping`、`simple-knn`（子模块）|
| Web 框架 | **无**，纯训练 / 推理脚本 |

> ⚠️ `environment.yml` 中标注的 `python=3.7.13` 与实际环境 3.9.25 不一致，以**实际环境 3.9.25** 为准。

---

## 3. 环境管理（铁律）

### 3.1 绝对铁律

1. **不主动安装任何依赖**。仅在用户明确要求「安装 X」时才执行安装命令。
2. **一律用 `pip`，禁止用 `conda install`**（避免污染 conda 频道、避免 yml 漂移）。
3. **安装目标固定为当前 conda 环境**：必须用 `conda run -n langsplat pip install ...` 或先 `conda activate langsplat` 再 `pip install`。
4. **装完必须同步更新 [environment.yml](file:///home/xiedexia/project/LangSplat/environment.yml)**（在 `pip:` 段追加），并告知用户已更新。
5. **禁止修改 conda 环境名**（固定为 `langsplat`），禁止新建环境。

### 3.2 Python 解释器

- 路径：`/home/xiedexia/.conda/envs/langsplat/bin/python`
- 版本：3.9.25
- 验证：`conda run -n langsplat python --version`

---

## 4. 常用命令（必须显式带 `conda run -n langsplat`）

> 所有命令在 `/home/xiedexia/project/LangSplat` 根目录执行；变量 `$DS` 表示数据集路径，`$CASE` 表示案例名。

### 4.1 预处理（SAM + CLIP 提取语言特征）

```bash
conda run -n langsplat python preprocess.py --dataset_path $DS
```

### 4.2 Autoencoder 训练与降维

```bash
cd autoencoder
conda run -n langsplat python train.py \
  --dataset_path $DS \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512 \
  --lr 0.0007 \
  --dataset_name $CASE

conda run -n langsplat python test.py \
  --dataset_path $DS \
  --dataset_name $CASE
cd ..
```

### 4.3 LangSplat 两阶段训练（核心）

```bash
# 前置：先用 3DGS 训练 RGB 模型（chkpnt30000.pth），不在本仓库职责内
# 阶段 2：在 RGB checkpoint 之上，对 level 1/2/3 各训练一次语言高斯
for level in 1 2 3; do
  conda run -n langsplat python train.py \
    -s $DS \
    -m output/${CASE}_${level} \
    --start_checkpoint $DS/$CASE/chkpnt30000.pth \
    --feature_level ${level}
done
```

### 4.4 渲染

```bash
# 渲染 RGB
conda run -n langsplat python render.py -m output/${CASE}_${level}

# 渲染语言特征图
conda run -n langsplat python render.py -m output/${CASE}_${level} --include_feature

# QuickStart（用预训练模型直接推理）
conda run -n langsplat python render.py -m output/$CASE --include_feature
```

### 4.5 评估

```bash
cd eval
sh eval.sh   # 内部调用 evaluate_iou_loc.py
cd ..
```

### 4.6 文本查询

```bash
conda run -n langsplat python query.py \
  --model_path output/${CASE} \
  --ae_ckpt autoencoder/ae_ckpt/${CASE}.pth \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512
```

### 4.7 数据转换（COLMAP）

```bash
conda run -n langsplat python convert.py -s $DS --camera SIMPLE_PINHOLE
```

### 4.8 常用辅助

```bash
# 检查环境
conda run -n langsplat python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 查看大文件
du -sh output/* dataset/* 2>/dev/null | sort -h
```

---

## 5. 目录结构（含大目录与「禁止读」标识）

> 🚫 = **禁止 AI 主动读取**（体积过大或为二进制权重，读取浪费上下文，仅在用户明确要求时按行范围 / 单文件读取）

### 5.1 代码与子模块目录（可读）

| 目录 | 作用 |
|---|---|
| [scene/](file:///home/xiedexia/project/LangSplat/scene) | 高斯模型、场景加载、相机、COLMAP/Blender 数据读取 |
| [gaussian_renderer/](file:///home/xiedexia/project/LangSplat/gaussian_renderer) | 高斯光栅化渲染（扩展支持语言特征渲染） |
| `autoencoder/` | CLIP 特征降维自编码器。**代码可读**（`train.py`/`model.py`/`test.py`），但 `autoencoder/ae_ckpt/*.pth` 禁止读（约 128 MB 权重）|
| [utils/](file:///home/xiedexia/project/LangSplat/utils) | 损失、SH、图像、系统工具 |
| [arguments/](file:///home/xiedexia/project/LangSplat/arguments) | 命令行参数定义 |
| [eval/](file:///home/xiedexia/project/LangSplat/eval) | 评估脚本与可视化 |
| [lpipsPyTorch/](file:///home/xiedexia/project/LangSplat/lpipsPyTorch) | LPIPS 模块 |
| [submodules/segment-anything-langsplat/](file:///home/xiedexia/project/LangSplat/submodules/segment-anything-langsplat) | SAM 分割模型 |
| [submodules/langsplat-rasterization/](file:///home/xiedexia/project/LangSplat/submodules/langsplat-rasterization) | 自定义高斯光栅化 CUDA 扩展 |
| [submodules/simple-knn/](file:///home/xiedexia/project/LangSplat/submodules/simple-knn) | KNN 用于初始点云 |

### 5.2 大目录（🚫 禁止主动读）

| 目录 | 体积 | 说明 | 标识 |
|---|---|---|---|
| `output/` | ~115 GB | 训练产物：`point_cloud.ply`、`chkpnt*.pth`、`cameras.json`、渲染结果 | 🚫 |
| `dataset/` | ~29 GB | 3D-OVS / LERF 数据集，含 `images/`、`sparse/`、`label/` | 🚫 |
| `output_backup/` | ~8.8 GB | 训练产物历史备份 | 🚫 |
| `ckpts/` | ~2.4 GB | SAM 预训练权重 | 🚫 |
| `ckpt/` | ~1.8 GB | 其他模型权重 | 🚫 |
| `autoencoder/ae_ckpt/` | ~128 MB | AutoEncoder 训练权重 | 🚫 |
| `query_results/` | ~101 MB | 查询结果图像 | 🚫 |
| `eval_result/` | — | 评估结果（基线）| 🚫 |
| `eval_result_final/` | — | 评估结果（最终版）| 🚫 |
| `eval_result_fixed/` | — | 评估结果（修复版）| 🚫 |
| `eval_result_fixed_new/` | — | 评估结果（修复新版）| 🚫 |
| `eval_result_new/` | — | 评估结果（新版）| 🚫 |
| `eval_result_new_final/` | — | 评估结果（新最终版）| 🚫 |
| `eval_result_official_ae/` | — | 评估结果（官方 AE）| 🚫 |
| `eval_result_sofa/` | — | 评估结果（sofa 场景）| 🚫 |
| `eval_result_test/` | — | 评估结果（测试）| 🚫 |
| `eval_result_v3/` | — | 评估结果（v3）| 🚫 |
| `eval_result_v3_final/` | — | 评估结果（v3 最终）| 🚫 |
| `eval_result_v3_test/` | — | 评估结果（v3 测试）| 🚫 |

### 5.3 大文件类型（🚫 禁止直接 Read）

- `.pth` —— 训练 checkpoint（最大约 **547 MB**）
- `.ply` —— 高斯点云 / 初始点云（最大约 **398 MB**）
- `.bin` —— COLMAP 二进制（cameras/images/points3D）
- `.npy` —— 渲染特征图 / GT 特征

读取这些二进制文件时，**改用** `torch.load(..., map_location='cpu')` 或 `numpy.load(..., mmap_mode='r')`，并在脚本里取切片。

**submodules/ 下的预训练权重也禁止读**，包括：
- `submodules/segment-anything-langsplat/*.pth`（SAM 权重，可能约 2.4 GB）
- `submodules/segment-anything-langsplat/**/*.pth`
- `submodules/segment-anything-langsplat/weights/`
- `submodules/segment-anything-langsplat/checkpoints/`

如发现新的子模块权重文件，及时补充到本清单和 .traeignore。

### 5.4 根目录入口脚本

| 脚本 | 作用 |
|---|---|
| [preprocess.py](file:///home/xiedexia/project/LangSplat/preprocess.py) | SAM + CLIP 提取 512 维语言特征 |
| [train.py](file:///home/xiedexia/project/LangSplat/train.py) | LangSplat 训练主入口 |
| [render.py](file:///home/xiedexia/project/LangSplat/render.py) | 渲染 RGB / 语言特征图 |
| [query.py](file:///home/xiedexia/project/LangSplat/query.py) | 文本查询入口 |
| [convert.py](file:///home/xiedexia/project/LangSplat/convert.py) | COLMAP 数据转换 |
| [process.sh](file:///home/xiedexia/project/LangSplat/process.sh) | 完整流水线示例脚本 |

> 根目录存在大量 `eval_*.py` / `debug_*.py` / `diagnose_*.py` 实验脚本，属于历史调试产物，修改前先确认是否仍被使用。

---

## 6. 编码规范

### 6.1 Python 版本（3.9，严禁 3.10+ 语法）

✅ **正确写法**：

```python
from typing import Optional, List, Dict, Tuple, Union

def foo(x: Optional[str] = None) -> List[int]:
    items: Dict[str, int] = {}
    arr: Tuple[int, ...] = (1, 2, 3)
    val: Union[str, int] = "a"
```

❌ **错误写法（3.10+ 语法，禁用）**：

```python
def foo(x: str | None = None) -> list[int]:   # 禁用 | Union 语法
    items: dict[str, int] = {}                 # 禁用小写内置泛型
    val: str | int = "a"                       # 禁用
```

### 6.2 其他规范

- 字符串格式化优先用 f-string；拼接长路径用 `os.path.join` / `pathlib.Path`。
- 路径处理优先 `pathlib.Path`，但避免依赖 3.10 才有的 `Path.is_relative_to` 等新 API。
- numpy / torch 操作显式写 `.cpu().numpy()` 转换，避免隐式 CUDA→CPU 转换报警告。
- 注释、docstring、print 信息**使用与用户消息相同的语言**（中文）。
- 不要给未改动的代码补 docstring / type hint / 注释；不要做无关重构。

### 6.3 PyTorch / numpy 规范

- torch tensor 不要直接 print，先 `.cpu().detach().numpy()` 再 print
- 涉及 CUDA 操作前显式 `torch.cuda.synchronize()`，避免计时不准
- 不要在循环里频繁 `.item()`，会同步 GPU 拖慢训练
- 保存 checkpoint 用 `torch.save({...}, path)`，不要直接 save 整个 model
- 加载 checkpoint 必加 `map_location='cpu'`，避免跨机器 CUDA 设备号问题
- 跨设备传 tensor 前 `.to(device)` 显式指定，不要依赖默认设备

### 6.4 文件 IO 规范

- 读 .npy 用 `np.load(path, mmap_mode='r')` 避免全量加载
- 读 .pth 用 `torch.load(path, map_location='cpu')`；如果 PyTorch 版本 ≥ 2.0，可加 `weights_only=True` 提升安全性；当前 PyTorch 1.12.1 **不支持** `weights_only` 参数，加了会报 `TypeError`
- 写中间结果到 `output/<case>/` 下，不要散落到项目根目录
- 大文件读写后显式 `del` + `gc.collect()`，避免显存泄漏
- 写大文件前检查磁盘空间：`df -h .`

### 6.5 实验可复现性

- 训练脚本必须支持 `--seed` 参数
- 关键实验记录 git commit hash + 命令行参数到 `output/<case>/config.json`
- 改代码后跑同一 seed 同一参数，确保结果可复现
- 不可复现的实验（涉及随机数据加载）必须固定 `torch.manual_seed()` 和 `np.random.seed()`

---

## 7. 训练实验管理规范

### 7.1 输出目录命名

- 输出目录：`output/${CASE}_${level}`，如 `output/sofa_3`
- `level` 必须显式（1/2/3），不能用 `output/sofa` 这种无 level 命名

### 7.2 checkpoint 路径

- `--start_checkpoint` 指向**已训练好的 RGB 3DGS 模型**（`chkpnt30000.pth`），不能省略
- checkpoint 路径示例：`$DS/$CASE/chkpnt30000.pth`

### 7.3 两阶段顺序

- **阶段 A**：先用 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 训练 RGB 模型（30k iter）
- **阶段 B**：在本仓库 `train.py --include_feature` / `--feature_level` 下从 checkpoint 恢复并训语言特征
- 训练代码会校验 checkpoint 是否包含 12 项参数，若已含 feature 则 `first_iter=0` 重置（见 [train.py](file:///home/xiedexia/project/LangSplat/train.py)）

### 7.4 新实验前的检查

- 先跑 `du -sh output/${CASE}*` 确认不会覆盖既有产物
- 重要 checkpoint 先备份：`cp -r output/${CASE} output_backup/${CASE}_$(date +%Y%m%d_%H%M%S)`

### 7.5 迭代数约定

- 默认 30k / 60k
- 根目录 `eval_bed_60k.py` 等 `_60k` 后缀脚本对应 60k 迭代评估
- 实验脚本与迭代数对应关系：
  - `eval_*.py` → 30k 默认评估
  - `eval_*_60k.py` → 60k 评估
  - `eval_*_final.py` → 最终版评估

### 7.6 实验脚本归档

- 根目录散落的 `eval_*.py` / `debug_*.py` / `diagnose_*.py` 若已废弃，应统一移入 `experiments_archive/`
- 移动前在 PR 说明中列出
- **改前先问用户**，不要主动归档

### 7.7 Git 工作流

#### 分支策略
- `main` - 稳定分支，只接受通过验证的实验代码
- `develop` - 日常开发分支
- `feature/xxx` - 新功能分支（如 `feature/add-multiscale-eval`）
- `experiment/xxx` - 实验分支（如 `experiment/lr-tuning`）

#### Commit 规范
- 用 conventional commits：`feat:` / `fix:` / `refactor:` / `experiment:` / `docs:`
- 实验脚本 commit 加 `experiment:` 前缀，方便后续清理
- 一个 commit 一个目的，不要混合多个改动

#### 禁止操作
- 不要在 main 分支直接 commit
- 不要 git push --force
- 不要 commit 大文件（.pth / .ply / .npy），用 .gitignore 排除

#### 迁移准备
- 实验室 → 云服务器迁移前，确保所有代码都已 push 到远程
- environment.yml 必须是最新的
- data/ output/ 不进 git，单独迁移

### 7.8 GPU 资源协作

#### 启动训练前必查

```bash
# 看 GPU 占用
nvidia-smi

# 看是否有别人的训练进程
nvidia-smi | grep python

# 看自己的进程
ps aux | grep -E "python.*train" | grep -v grep
```

#### GPU 选择规范

- 不要默认用 GPU 0（别人最常用）
- 显式指定 GPU：`CUDA_VISIBLE_DEVICES=<N> conda run -n langsplat python train.py ...`
- 多卡训练前先在 Chat 里问我有没有占用
- 训练前先跑 100 iter smoke test，确认不 OOM 再跑完整训练

#### 训练中断处理

- 看到 CUDA out of memory 不要直接重试，先 `nvidia-smi` 看是不是被别人占了
- 长训练（>6 小时）用 nohup 或 tmux 跑：

```bash
tmux new -s train_sofa
CUDA_VISIBLE_DEVICES=1 conda run -n langsplat python train.py ...
# Ctrl+B D 脱离 tmux
# 回来：tmux attach -t train_sofa
```

- 训练被意外 kill 时，用 `ls -lt output/<case>/chkpnt*.pth | head` 列出所有 checkpoint 文件名（按修改时间排序），找到最新 chkpnt 编号，用 `--start_checkpoint <path>` 恢复训练。**注意：只 `ls` 列文件名，不要 `Read` 读取 .pth 文件内容（§8 第 7 条禁止读二进制权重）**。

### 7.9 改训练代码的标准流程

#### 改 loss / 模型结构前

1. 先 commit 当前状态：`git add -A && git commit -m "before: modify loss function"`
2. 跑 baseline 实验：用当前代码跑一次小训练（1000 iter），记录关键指标
3. 改代码
4. 跑 modified 实验：同样 1000 iter，对比指标
5. 如果变好：commit 改动，跑完整训练验证
6. 如果变差：`git checkout train.py` 回滚，分析原因

#### 改超参数前

- 不要在 main 分支改超参
- 用单独的 experiment 分支
- 记录每次实验的：超参值、最终指标、训练时长、GPU 占用

#### 改完后必须做

- 跑 1000 iter smoke test 验证 forward/backward 不报错
- 跑完整训练后跑评估，对比 baseline
- 把实验结果记到 `experiments_log.md`（在项目根目录创建）

---

## 8. 禁止操作清单

🚫 以下操作**未经用户明确同意禁止执行**：

1. 读取 / 删除 `output/`、`dataset/`、`output_backup/`、`ckpts/`、`ckpt/`、`autoencoder/ae_ckpt/`、`eval_result*/`、`query_results/` 下任何文件。（含 `autoencoder/ae_ckpt/` 下的 .pth 权重，与 §5.2 一致）
2. `rm -rf`、`git clean -fd`、`git reset --hard`、`git checkout .`、`git push --force`。
3. 修改 `environment.yml` 中的 conda 频道或 `python` / `pytorch` 版本。
4. `conda install`（环境管理铁律：只能 `pip`）。
5. 在未经用户确认时 `git commit` / `git push`。
6. 主动 `pip install` 任何包（铁律 1）。
7. 直接 `Read` 二进制权重（`.pth` / `.ply` / `.bin` / `.npy`）。
8. 修改 `submodules/` 内的子模块源码（除非用户明确要求且理解后果）。
9. 修改 `arguments/__init__.py` 的默认参数（影响所有训练命令）。

---

## 9. 已知问题与注意事项

### 9.1 CUDA / 显存
- 训练需 24 GB VRAM；低于此显存会 OOM。
- `cudatoolkit=11.6` 与 PyTorch 1.12.1 强绑定，换 CUDA 版本需重新编译 `submodules/langsplat-rasterization` 与 `submodules/simple-knn`。
- 若光栅化扩展报 `ImportError: ... undefined symbol`，通常是 CUDA 版本不匹配 → 重装：`conda run -n langsplat pip install submodules/langsplat-rasterization --force-reinstall --no-deps`。

### 9.2 数据路径
- `--source_path` / `-s` 指向数据集根，其下必须含 `images/`、`sparse/0/{cameras,images,points3D}.bin`。
- 语言特征目录名默认 `language_features_dim3`（由 [arguments/__init__.py](file:///home/xiedexia/project/LangSplat/arguments/__init__.py) 的 `_language_features_name` 控制）。
- 3D-OVS 数据集在 `dataset/3D Open-vocabulary Segmentation datasets/`（注意目录名带空格，shell 命令必须加引号）。
- LERF 数据集在 `dataset/lerf_ovs/`，标签在 `dataset/lerf_ovs/label/`。

### 9.3 3DGS 两阶段训练
- **阶段 1（RGB）** 不在本仓库，需先在原版 3DGS 训练得到 `chkpnt30000.pth`。
- **阶段 2（语言）** 在本仓库 `train.py`，从阶段 1 的 checkpoint 恢复，再训 `feature_level` 1/2/3 三次。
- `feature_level`：1=粗、2=中、3=细；最终评估用 level 3（或级联取最高置信度）。
- 若 checkpoint 已含 feature 参数（`len(model_params)==12`），训练会重置 `first_iter=0`，避免继续迭代导致过拟合。

### 9.4 Autoencoder
- 默认 `encoder_dims=256 128 64 32 3`、`decoder_dims=16 32 64 128 256 256 512`。
- 中间层在编码后做 L2 归一化，解码后再次归一化（见 [autoencoder/model.py](file:///home/xiedexia/project/LangSplat/autoencoder/model.py)）。
- `eval_result_official_ae/` 对应使用官方预训练 AE 的评估结果，与本地训练的 AE 区别开。

### 9.5 实验脚本污染（具体清单）

根目录有大量历史调试脚本，按类型分类：

#### Eval 脚本变体（场景×版本）

- `eval_sofa.py` / `eval_sofa_correct.py` / `eval_sofa_correct2.py`
- `eval_sofa_fixed.py` / `eval_sofa_optimized.py` / `eval_sofa_optimized2.py`
- `eval_bed.py` / `eval_bed_60k.py` / `eval_bed_final.py`
- `eval_new.py`
- 对应评估结果目录：`eval_result_sofa/`、`eval_result_fixed/`、`eval_result_new/` 等。

#### Debug 脚本（一次性诊断）

- `debug_eval.py` / `debug_eval2.py` / `debug_eval3.py`
- `debug_final.py` / `debug_final2.py`
- `debug_gt.py` / `debug_gt2.py`

#### Diagnose 脚本（问题定位）

- `diagnose_eval.py` / `diagnose_eval2.py`
- `diagnose_issue.py` / `diagnose_issue_v2.py` / `diagnose_issue_v3.py`

#### 转换脚本

- `convert_sofa_segmentations.py` / `convert_sofa_segmentations_v2.py` / `convert_sofa_segmentations_v3.py`
- `encode_bed_dim3.py` / `reencode_with_sofa_ae.py`

#### 处理原则

- 修改前先问用户：明确告诉用户"我要改 `eval_sofa_optimized2.py`，但发现还有 `eval_sofa_optimized.py`，哪个是你想要的？"
- 不要假设最新版本是正确版本：v3 可能不如 v2
- 废弃脚本归档：用户确认废弃后，统一移到 `experiments_archive/` 目录
- 新建脚本命名：用 `eval_<scene>_<purpose>.py` 格式，避免数字后缀堆叠

---

## 10. 调试技巧

### 10.1 渲染 / 特征异常
- **现象**：`render.py --include_feature` 输出全黑 / NaN。
- **排查**：
  1. 确认 `chkpnt*.pth` 中 `language_feature` 字段存在（`torch.load(..., map_location='cpu')` 后看 keys）。
  2. 确认 `language_features_dim3/` 下 `.npy` 文件齐全且维度=3。
  3. 对照 [render.py](file:///home/xiedexia/project/LangSplat/render.py) 中 `output["language_feature_image"]` 分支，检查 `args.include_feature` 是否为 True。
  4. 已有诊断脚本：[diagnose_eval.py](file:///home/xiedexia/project/LangSplat/diagnose_eval.py)、[diagnose_issue.py](file:///home/xiedexia/project/LangSplat/diagnose_issue.py)、[debug_gt.py](file:///home/xiedexia/project/LangSplat/debug_gt.py)、[debug_final.py](file:///home/xiedexia/project/LangSplat/debug_final.py)（先读再跑）。

### 10.2 评估 IoU 偏低
- 检查 GT 标签路径 `--gt_folder` 是否指向 `dataset/lerf_ovs/label/`。
- 检查 Autoencoder checkpoint 是否与训练时 `encoder_dims/decoder_dims` 一致。
- 参考 [analyze_feature_quality.py](file:///home/xiedexia/project/LangSplat/analyze_feature_quality.py) 检查特征余弦相似度分布。

### 10.3 OOM
- 降低分辨率：`--resolution 2` 或 `-r 2`。
- 减少 batch / 减少 SfM 图像数。
- 用 `nvidia-smi` 确认无残留进程；必要时 `conda run -n langsplat python -c "import torch; torch.cuda.empty_cache()"`。

### 10.4 调试运行
- 单步断点：`conda run -n langsplat python -m pdb train.py ...`
- TensorBoard：`conda run -n langsplat tensorboard --logdir output --port 6006`。
- 快速 smoke test：用 `test_pipeline.sh`（先读再跑）。

### 10.5 日志与产物
- 训练日志在 `output/${CASE}_${level}/`（TensorBoard 事件、`cameras.json`、`cfg_args`）。
- 评估日志在对应 `eval_result_*/` 目录的 `metrics.json`。

---

## 11. 迁移到云服务器的准备

> 本节是更全面的迁移准备清单。Git 角度的简短迁移准备见 §7.7。

### 11.1 机器相关配置（迁移时必改）

以下配置含实验室机器的绝对路径或硬件特征，迁移到云服务器时必须更新：

#### 路径相关

- 所有 `--source_path` / `-s` 参数指向 `/home/xiedexia/...`，迁移后要改成云服务器的用户路径（如 `/home/deploy/...`）
- `--start_checkpoint` 指向的 RGB checkpoint 路径
- `dataset/` 内的数据集路径（如果用软链接）
- TensorBoard logdir

#### 硬件相关

- `CUDA_VISIBLE_DEVICES` 设置（云服务器可能只有 1 张卡）
- batch size（云服务器显存可能不同）
- `--resolution` 参数（云服务器可能不需要降分辨率）

#### 环境相关

- conda 环境路径：`/home/xiedexia/.conda/envs/langsplat/` → 云服务器路径
- Python 3.9.25 → 云服务器建议装同样版本
- PyTorch + CUDA 版本 → 云服务器 GPU 不同可能要换 CUDA 版本
- `submodules/langsplat-rasterization` 和 `submodules/simple-knn` 需要重新编译

### 11.2 迁移前必做的检查清单

- [ ] 所有代码 commit 到 git，无未推送改动
- [ ] `environment.yml` 是最新的（`conda env export --no-builds > environment.yml`）
- [ ] `.traeignore` 和 `.trae/rules/project_rules.md` 已 commit
- [ ] 大文件（`output/` `dataset/` `ckpts/`）单独 rsync 迁移，不走 git
- [ ] 实验脚本归档到 `experiments_archive/`，减少迁移负担
- [ ] 记录当前所有训练实验的命令行参数和 checkpoint 路径
- [ ] 验证：在云服务器 clone 项目后，能从 `environment.yml` 重建环境、跑通 smoke test

### 11.3 不该迁移的内容

- `output/` 115 GB 训练产物：除非要继续训练，否则不迁移（重新训练更省事）
- `output_backup/` 8.8 GB 历史备份：不迁移
- `eval_result_*/` 评估结果：除非要对比，否则不迁移
- `__pycache__/` 缓存：不迁移

### 11.4 迁移步骤参考

详细迁移步骤参见项目根目录上级的 `trae_manual1_1.md` 第 11 章（场景三：实验室 → 云服务器迁移）。本节仅做规则约束，具体操作以该手册为准。

---

## 附：规则维护

- 本文件位于 [.trae/rules/project_rules.md](file:///home/xiedexia/project/LangSplat/.trae/rules/project_rules.md)，由 Trae AI 在本仓库工作时强制遵循。
- 修改本文件前先与用户确认；新增大目录 / 大文件时及时补充到 §5。
- 与 [environment.yml](file:///home/xiedexia/project/LangSplat/environment.yml)、[process.sh](file:///home/xiedexia/project/LangSplat/process.sh)、[README.md](file:///home/xiedexia/project/LangSplat/README.md) 保持一致。

### 版本历史

**v1.1（2026-07）：**
- 修正 §5.1 `autoencoder/` 排除矛盾（拆分代码可读 + 权重禁止读）
- §5.3 补充 submodules 下大权重排除规则
- §6 新增 PyTorch/numpy 规范、文件 IO 规范、实验可复现性
- §7 结构改造：原列表项拆为 §7.1-§7.6 子标题；新增 §7.7 Git 工作流、§7.8 GPU 资源协作、§7.9 改训练代码标准流程
- §8 禁止操作清单同步补齐 `autoencoder/ae_ckpt/`
- §9.5 实验脚本污染改为具体清单版
- 新增 §11 迁移到云服务器的准备

**v1.0（2026-07）：**
- 初版：项目简介、技术栈、环境管理、常用命令、目录结构、编码规范、训练实验管理、禁止操作、已知问题、调试技巧
