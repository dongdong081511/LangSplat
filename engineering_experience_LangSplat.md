# LangSplat 工程经验总结

> 本文档记录 LangSplat 多模态融合实验中积累的工程经验，避免重复试错。
> 最后更新: 2026-09-19 (分支: experiment/crossattn-mm)

---

## 1. 环境与运行约束

### 1.1 Conda 环境与 Python 路径
- 环境名: `langsplat`，路径: `/home/xiedexia/.conda/envs/langsplat`
- **必须用** `/home/xiedexia/.conda/envs/langsplat/bin/python -u` 直接调用
- **不要用** `conda run -n langsplat python`，会缓冲 stdout 导致后台任务看不到输出
- `-u` 参数禁用输出缓冲，确保后台任务日志实时写入

### 1.2 CUDA 与沙箱 (Sandbox)
- **所有 CUDA 训练/渲染命令必须在 Shell 工具中设置 `dangerouslyDisableSandbox: true`**
- 否则沙箱阻止 `/proc/<pid>/task/<tid>/comm` 访问，导致 `CUDA Error 304: OS call failed`
- 编译 rasterizer 也需要禁用沙箱 (需访问 NVCC、g++)

### 1.3 .gitignore 必须排除的大目录
```
dataset/
ckpts/
eval_result/
output_backup/
output
*.log
ckpt
```
- `ckpt` (无路径前缀) 会匹配任意层级的 `ckpt/` 目录，包括 `mm_langsplat/ckpt/` 和 `autoencoder/ckpt/`

---

## 2. 多模态融合 Pipeline 完整流程

### 2.1 Pipeline 步骤 (顺序依赖)
```
1. 预处理 (extract CLIP+DINOv2) → 2. 融合网络训练 → 3. 生成 fused 512d 特征
→ 4. AE 训练 (512→N) → 5. encode_dim3 (生成 N d 特征) 
→ 6. 3-level Gaussian 训练 → 7. render → 8. eval 评估
```

### 2.2 各步骤命令参考 (以 teatime 12d 为例)

**步骤1: 预处理提取 CLIP+DINOv2 特征**
```bash
python -u preprocess.py --dataset_name teatime --images images \
  --scale 2 --dino --sam
```
- 输出: `dataset/lerf_ovs/teatime/language_features/` (CLIP _f.npy)
- 多模态输出: `dataset/lerf_ovs/teatime/language_features_mm/` (CLIP _f.npy + DINOv2 _f_dino.npy)

**步骤2: 训练融合网络**
```bash
python -u -m mm_langsplat.train_fusion \
  --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
  --save_path mm_langsplat/ckpt/fusion_teatime.pth \
  --epochs 200 --lr 1e-4 --batch_tiles 256 \
  --loss_type {mse_cos,mse_only,infonce} --temperature 0.07
```

**步骤3: 生成 fused 512d 特征**
```bash
python -u -m mm_langsplat.gen_fused_features \
  --feat_dir dataset/lerf_ovs/teatime/language_features_mm \
  --out_dir dataset/lerf_ovs/teatime/language_features_fused \
  --ckpt_path mm_langsplat/ckpt/fusion_teatime.pth
```

**步骤4: AE 训练 (512→N)**
```bash
cd autoencoder && python -u train.py \
  --dataset_path ../dataset/lerf_ovs/teatime \
  --data_subdir language_features_fused \
  --dataset_name teatime_fused_12d \
  --encoder_dims 256 128 64 32 12 --decoder_dims 16 32 64 128 256 256 512
```
- ckpt 保存到 `autoencoder/ckpt/<dataset_name>/best_ckpt.pth`
- **必须复制到 ae_ckpt 子目录** (eval 脚本需要): `cp best_ckpt.pth ae_ckpt/best_ckpt.pth`

**步骤5: encode_dim3 生成 N d 特征**
```bash
python -u -m mm_langsplat.encode_dim3 \
  --dataset_path dataset/lerf_ovs/teatime \
  --data_subdir language_features_fused \
  --out_subdir language_features_dim12_fused \
  --ae_ckpt autoencoder/ckpt/teatime_fused_12d/best_ckpt.pth \
  --encoder_dims 256 128 64 32 12 --decoder_dims 16 32 64 128 256 256 512
```

**步骤6: 3-level Gaussian 训练**
```bash
python -u train.py -s dataset/lerf_ovs/teatime \
  -m output/teatime_fused_12d \
  --start_checkpoint dataset/lerf_ovs/teatime/output/teatime_-1/chkpnt30000.pth \
  --language_features_name language_features_dim12_fused \
  --feature_level <1|2|3> --include_feature --port <55571|55572|55573>
```

**步骤7: render**
```bash
python -u render.py -m output/teatime_fused_12d_<level> \
  --include_feature \
  --language_features_name language_features_dim12_fused --feature_level <level>
```
- 输出: `output/<name>_<level>/train/ours_None/renders_npy/` (177 个 npy)

**步骤8: eval**
```bash
cd eval && python -u evaluate_iou_loc.py \
  --feat_dir ../output --dataset_name teatime_fused_12d \
  --ae_ckpt_dir ../autoencoder/ckpt \
  --json_folder ../dataset/lerf_ovs/label --output_dir ../eval_result \
  --mask_thresh 0.4 \
  --encoder_dims 256 128 64 32 12 --decoder_dims 16 32 64 128 256 256 512
```
- **eval 必须从 `eval/` 子目录运行**，否则相对路径解析错误

---

## 3. 关键代码修改 (多维度支持)

### 3.1 修改 rasterizer 维度
- 文件: `submodules/langsplat-rasterization/cuda_rasterizer/config.h`
- 改 `NUM_CHANNELS_language_feature` 为目标维度 (如 12)
- **必须重新编译**:
```bash
cd submodules/langsplat-rasterization && pip install -e .
```
- 编译环境变量:
```bash
export CUDA_HOME=/home/xiedexia/.conda/envs/langsplat
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
```

### 3.2 需要同步修改维度的文件
| 文件 | 修改内容 |
|------|---------|
| `submodules/langsplat-rasterization/cuda_rasterizer/config.h` | `NUM_CHANNELS_language_feature` |
| `scene/gaussian_model.py` | `_language_feature` 的 `torch.zeros((N, 12))` |
| `gaussian_renderer/__init__.py` | `language_feature_precomp` 的 `torch.zeros((N, 12))` |
| `eval/evaluate_iou_loc.py` | `compressed_sem_feats` 用 `encoder_hidden_dims[-1]` 替代硬编码 3 |
| `render.py` | 仅 `rendering.shape[0] <= 3` 时保存 PNG (多通道特征非 RGB) |

### 3.3 其他关键修改
- `autoencoder/train.py`: 添加 `--data_subdir` 参数 (原硬编码 `language_features`)
- `train.py`: 处理 13 参数 checkpoint (含 language_feature 的 restore)
- `utils/general_utils.py`: `safe_state()` 中 `torch.cuda.set_device` 加 try/except 防沙箱崩溃

---

## 4. 融合网络架构与训练经验

### 4.1 架构: TileCrossAttentionFusion
- **tile 级** cross-attention (非像素级)，避免 OOM
  - 像素级: 480×640=307200 tokens, attention ~376GB → OOM
  - tile 级: ~300 tokens, O(300²) → 安全
- 结构: proj→self-attn→cross-attn→FFN→proj_out，输出 512d (CLIP 兼容)
- CLIP input dropout (0.3) 强制学习 DINOv2 特征

### 4.2 关键训练约束
- **网络初始化必须随机初始化，禁止 zero init** (否则 loss=0，网络只复制 CLIP)
- **融合网络训练时需移除 residual 连接** (否则网络仅复制 CLIP 而未学习 DINOv2)
- 训练 200 epochs，cos_sim 收敛到 ~0.98 (mse_cos/mse_only) 或 ~0.58 (infonce)

### 4.3 损失函数对比 (12d, teatime)
| 损失函数 | 融合 cos_sim | AE loss | IoU | Loc |
|---------|-------------|---------|-----|-----|
| mse_cos (原版) | 0.9806 | 0.0922 | 0.6609 | 0.8644 |
| **mse_only (最优)** | 0.9796 | 0.0843 | **0.6723** | **0.9153** |
| infonce | 0.5810 | 0.2412 | 0.5270 | 0.8475 |

- **MSE-only 最优**: cos_sim 项对 unit-normalized 输出与 MSE 近似等价 (MSE ≈ 2*(1-cos_sim))，属冗余约束
- **InfoNCE 失败**: 对比学习使 fused 过度偏离 CLIP 空间，AE 重建困难，下游 IoU 大幅下降

---

## 5. AE 瓶颈实验发现

| 方案 | AE dim | IoU | Loc | 关键结论 |
|------|--------|-----|-----|---------|
| baseline | 3 | 0.6431 | 0.8983 | CLIP 最优维度是 3d |
| baseline | 6 | 0.6285 | 0.8644 | 纯 CLIP 在 6d 反而下降 |
| baseline | 12 | 0.6660 | 0.8814 | 12d 是 CLIP 更优维度 |
| fused (mse_cos) | 3 | 0.5351 | 0.8814 | 3d 瓶颈严重损害 fused |
| fused (mse_cos) | 6 | 0.6251 | 0.9153 | 6d 对 fused 有选择性增益 |
| fused (mse_cos) | 12 | 0.6609 | 0.8644 | 12d 几乎追平 baseline |
| **fused (mse_only)** | **12** | **0.6723** | **0.9153** | **首次超越 baseline** |
| baseline | 16 | 0.6401 | 0.8475 | baseline 超过 12d 后退化 |
| fused (mse_only) | 16 | 0.6698 | 0.8814 | fused 饱和略降，仍超 baseline +3.0% |

- **3d 瓶颈是不成比例损害 fused 特征的根本原因**
- fused IoU 随维度单调上升至 12d 后饱和: 3d 0.5351 → 6d 0.6251 → 12d 0.6609/0.6723 → 16d 0.6698
- baseline 在 12d 达峰后退化: 6d 0.6285 → 3d 0.6431 → 12d 0.6660 → 16d 0.6401
- **fused 对 AE 维度更鲁棒**: 12d→16d，baseline 退化 -2.6pp，fused 仅 -0.25pp，fused 优势扩大至 +3.0pp IoU / +3.4pp Loc
- **DINOv2 融合增益依赖: 损失函数设计 (MSE-only) + AE 维度 (12d) 共同作用**

---

## 6. 场景特定配置

| 场景 | mask_thresh | 图片数 |
|------|------------|--------|
| figurines | 0.45 | - |
| waldo_kitchen | 0.40 | - |
| teatime | 0.40 | 177 |
| ramen | 0.55 | - |

- **ramen 场景训练时不要用 `--eval` 标志**，否则 train/test split 导致渲染帧数不足，eval 时 IndexError

---

## 7. 并行训练端口分配

- train.py 默认 port 55555
- 并行训练用不同 `--port` 区分 (如 55571-55576)
- 6 个 Gaussian 训练可同时并行 (GPU 49GB 显存足够)

---

## 8. 常见坑与解决方案

### 8.1 CUDA Error 304
- **原因**: 沙箱阻止 `/proc/<pid>/task/<tid>/comm` 访问
- **解决**: Shell 工具设置 `dangerouslyDisableSandbox: true`

### 8.2 融合网络 loss=0 (不学习)
- **原因**: zero init + residual 连接 → 网络直接复制 CLIP
- **解决**: 随机初始化 + 去掉 residual

### 8.3 AE 重建好但下游 IoU 差
- **现象**: AE loss 低不代表下游任务好
- **原因**: 融合网络 cos_sim=0.98 说明主要在重建 CLIP，但丢失语义判别力
- **解决**: 用 MSE-only 损失 + 足够 AE 维度 (12d)

### 8.4 render.py 多通道特征保存失败
- **原因**: 12d 特征非 RGB 3 通道，`torchvision.utils.save_image` 失败
- **解决**: 加 `if rendering.shape[0] <= 3:` 守卫

### 8.5 gaussian_renderer 内存越界
- **原因**: `language_feature_precomp` 分配 3 元素，但 12d kernel 读取 12 元素
- **解决**: 分配 `torch.zeros((N, 12))` 匹配维度

### 8.6 后台任务看不到输出
- **原因**: `conda run` 缓冲 stdout
- **解决**: 直接用 `python -u` + `| tail -N` (输出在命令结束后写入日志)

---

## 9. 数据目录结构

```
dataset/lerf_ovs/teatime/
├── language_features/              # CLIP 512d (原始)
├── language_features_mm/            # CLIP + DINOv2 (_f.npy + _f_dino.npy)
├── language_features_fused/         # 融合后 512d
├── language_features_dim12/         # AE 编码后 12d (baseline)
├── language_features_dim12_fused/   # AE 编码后 12d (fused)
└── output/teatime_-1/              # 基线 Gaussian (chkpnt30000.pth 作为 start_checkpoint)

autoencoder/ckpt/
├── teatime_12d/                     # baseline 12d AE
│   ├── best_ckpt.pth
│   └── ae_ckpt/best_ckpt.pth        # eval 脚本用的副本
└── teatime_fused_12d/                # fused 12d AE

dataset/lerf_ovs/label/
├── teatime/                         # 真实标注
├── teatime_12d -> teatime            # 软链接 (eval 按 dataset_name 查找)
└── teatime_fused_12d -> teatime
```

- **label 软链接**: eval 脚本按 `--dataset_name` 在 `label/` 下查找同名目录，需创建软链接指向真实标注

---

## 10. 关键结论速查

1. **最优配置**: cross-attn 融合 + MSE-only 损失 + 12d AE → IoU=0.6723 (超 baseline +0.6%)；16d 已验证饱和 (EXP-011/012)，12d 为最优维度
2. **损失排序**: mse_only > mse_cos > infonce
3. **维度排序**: 12d > 16d > 6d > 3d (对 fused); 12d > 3d > 16d > 6d (对 baseline)
4. **DINOv2 有增益**: 但需 损失函数 + AE 维度 共同作用，三者缺一不可
5. **实验记录**: 所有参数和结果追加写入 `hyper_parameter.md`，不替换历史数据
