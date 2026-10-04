# LangSplat 工程经验总结

> 本文档记录 LangSplat 多模态融合实验中积累的工程经验，避免重复试错。
> 最后更新: 2026-09-21 (分支: experiment/crossattn-mm)

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
| **baseline** | **8** | **0.6705** | **0.8814** | **baseline 新最优（超过 12d，EXP-018）** |
| fused (mse_only) | 8 | 0.6375 | 0.8644 | fused 需 ≥12d，8d 明显退化 |

- **3d 瓶颈是不成比例损害 fused 特征的根本原因**
- fused IoU 随维度上升至 12d 后饱和: 3d 0.5351 → 6d 0.6251 → **8d 0.6375 (下限敏感)** → 12d 0.6609/0.6723 → 16d 0.6698
- **fused 对维度下限敏感、对上限鲁棒**: 12d→8d 降 3.5pp，12d→16d 仅降 0.25pp——融合特征信息量大，需要足够维度承载
- baseline 维度曲线 (EXP-018 修正): 6d 0.6285 → 3d 0.6431 → 16d 0.6401 → 12d 0.6660 → **8d 0.6705 (最优)**
- **fused 对 AE 维度更鲁棒**: 12d→16d，baseline 退化 -2.6pp，fused 仅 -0.25pp，fused 优势扩大至 +3.0pp IoU / +3.4pp Loc
- **DINOv2 融合增益依赖: 损失函数设计 (MSE-only) + AE 维度 ≥12d 共同作用**；8d 时 baseline 反超 fused（EXP-018）

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

### 8.7 语言特征维度硬编码 (改维度必须全改)
- **位置**: ① submodules/langsplat-rasterization/cuda_rasterizer/config.h NUM_CHANNELS_language_feature
  ② gaussian_renderer/__init__.py (~L93 language_feature_precomp zeros)
  ③ scene/gaussian_model.py (~L206 language_feature zeros)
- **现象**: config.h 改回 12 但 py 侧残留 16 → backward 梯度 [N,12] vs 期望 [N,16] 报错
- **解决**: 改维度时三处同步改，然后 `pip install -e . --no-build-isolation` 重编译

### 8.8 第三模态 tile 特征行数对齐
- **原因**: _f.npy 行数 = seg_map.max()+1（每帧不同 ~295-305，非固定 300）
- **现象**: 深度特征写死 [300,8] → valid mask 索引报 boolean dimension mismatch
- **解决**: 按 seg_map.max()+1 生成特征行数，加载后校验 shape 与 _f.npy 一致

### 8.10 辅助损失权重与 CLIP 兼容性的此消彼长 (EXP-015/016)
- **现象**: depth 辅助损失 λ=1.0 → cos_sim 0.86 → IoU 0.16 (崩塌); λ=0.1 → cos_sim 0.94 → IoU 0.56; λ=0 → cos_sim 0.98 → IoU 0.65
- **教训**: 凡把 fused 拉离 CLIP 空间的损失项 (InfoNCE / 强辅助损失) 都会灾难性损害下游开放词汇查询
- **验证方法**: 训练融合网络后先看 cos_sim，<0.95 基本注定下游失败，无需跑完 1 小时管线

### 8.11 后台编译与 config.h 并行修改的竞态
- **现象**: pip 重编译 (后台) 与 Edit config.h 并行执行，编译产物通道数不确定 → 训练 illegal memory access
- **规则**: 改 config.h 必须先确认写入成功 (grep 验证)，再启动编译；编译后用一次训练迭代验证通道数

### 8.9 pkill 误杀自身 shell
- **原因**: `pkill -f <模式>` 匹配包含该模式的自身 bash 命令行 → exit -1，后续启动命令未执行
- **解决**: 长训练用 `setsid nohup ... &` 脱离会话；避免 pkill 与启动放在同一条命令

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
├── language_features_fused3/        # 三模态融合后 512d (EXP-013)
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


### 8.12 AE 编码竞态：特征编码必须晚于 AE 训练收敛 (EXP-017)
- **现象**: figurines baseline 首次 eval IoU=0.0194 / Loc=0.0179 (随机水平)，且对 mask_thresh 扫描完全不敏感
- **根因**: `mm_langsplat/encode_dim3.py` 在 AE 训练尚未收敛时执行 (dim12 编码 18:14，AE best_ckpt 19:13 才最终保存)，3D GS 用过期权重编码的特征训练，全链条判别结构损毁
- **定位方法 (三源对照法，推荐)**: 对同一 eval 帧，用 eval 完全相同的 relevancy 路径 (get_max_across + softmax) 分别测：
  1. 原始 2D CLIP 特征 → figurines frame_00041 IoU=0.593 (特征本身健康)
  2. AE 往返 (过期编码文件 decode) → IoU=0.063 (崩溃在 AE 环节)
  3. 3D 渲染 decode → IoU=0.025 (与 eval 一致)
  - 用最终 ckpt 现场重编码再 decode → IoU=0.669，确认修复
- **关键陷阱**: 平均 cos_sim 无法发现此问题 (过期编码 decode cos=0.927 看似健康，但判别结构已毁——再次验证"平均重建好 ≠ 下游好")；必须用下游 relevancy/IoU 验证
- **预防规则**: AE 训练完成后，核对 `stat` 时间戳：encode 输出文件的 mtime 必须 > best_ckpt.pth 的 mtime，否则重跑 encode。AE 训练分两次跑 (两个 tfevents) 时尤其高危
- **辅助定位经验**: 崩溃排查顺序 = ①可视化叠加图确认视角对齐 (composited vs 原图) ②高 relevancy 查询 (pikachu IoU=0.977@2D) 证明特征健康 ③三源对照锁定环节

### 8.13 维度不匹配错误的双向诊断法 (EXP-022 教训)
- **报错格式**: `backward returned an invalid gradient at index 4 - got [N,8] but expected [N,24]`
- **语义**: `got` = backward 返回的 dL 分配 shape (**C++ 侧** NUM_CHANNELS_language_feature 编译值); `expected` = forward 输入 grad shape (**Python 侧** zeros 维度)
- **诊断规则**: 看 got 判断 C++ kernel 实际状态, 看 expected 判断 Python 侧状态——方向搞反会误判"编译没生效"
- **案例**: got [N,8]/expected [N,24] 时 C++ 已是 8d, 真凶是 python 侧未改——差点重复全量重编

### 8.14 pip install -e 重编译的三重坑 (EXP-022)
- **坑1 增量假编译**: setuptools 不可靠跟踪头文件 (config.h) 变更, .cu 未重编但 .so 时间戳更新 (25 秒"编译完成"), 产物可能是旧 kernel + 新 zeros 的混合状态 (forward/backward 模板 24d + rasterize_points zeros 8d), 训练报 illegal access 或形状错误
- **坑2 CUDA 304**: rm -rf build 后 torch 需运行时探测 GPU 架构触发 CUDA init → 沙箱 304。解决: `export TORCH_CUDA_ARCH_LIST="8.9"` 显式指定
- **坑3 editable .so 混乱**: PEP 660 editable wheel 产物位置不定 (build/lib vs 源码树), finder 映射可能与实际 .so 不一致
- **可靠编译流程**: `rm -rf build && python setup.py build_ext --inplace` (产物固定在源码树 diff_gaussian_rasterization/), 编译后必须 100s 冒烟训练验证维度

### 8.15 python 脚本 patch 文件的原子性
- **教训**: 一个脚本里多次 str.replace + assert, 若中途 assert 失败, 前面已修改的内存变量未写盘 → 部分 patch 静默丢失
- **解决**: 每次修改独立脚本独立写入; 或 assert 全部通过后再统一写入; 修改后必须 grep 验证
- **案例**: evaluate_iou_loc.py CLI patch 因 old_call assert 失败导致 CLI 参数丢失, eval 报 unrecognized arguments

### 8.16 SAM preprocess 不能双场景并行 (EXP-022)
- **现象**: teatime+figurines 两个 preprocess.py 同时启动, 双双卡在 0it (CPU 后处理争抢互卡)
- **解决**: preprocess 严格串行 (~19.4s/it 正常), GPU 训练任务才可并行

### 8.17 score 级融合的温度敏感性 (EXP-023)
- **现象**: 双流 score 融合 λ 单调劣化 (0.6705 → λ0.3 0.5536 → λ0.5 0.6107 非单调但均降), DINO 投影流为纯噪声
- **机制**: 对齐层高 cos_sim (0.9256) ≠ 判别力——整体相似只保证投影落进 CLIP 空间, 不保证 tile 间区分; get_relevancy 的 softmax(10·sims) 温度下, 弱判别流的分布平坦, 融合后稀释 CLIP 尖峰
- **验证方法**: 注入新模态前先检查投影特征的 tile 间相似度方差 (判别性指标), 方差接近 0 的流注入必失败, 无需跑 2 小时管线
- **评估协议注意**: 换 dataset_name 做 λ 扫描时 feat_dir/output/label 三处目录名联动, 需要 renders 软链接 + label 软链接配套, 否则 IndexError/UnboundLocalError

### 8.18 tile 级 mean-pool 会抹掉 dense 特征的空间选择性 (EXP-025)
- **现象**: MaskCLIP 式 dense 特征 (末层 v-projection) 对每个 SAM tile 内 ~196 个 patch 取 mean 后, 2D tile 级 relevancy IoU 从 0.44/0.54 崩到 0.03 (-40pp), 无论是否模板 ensemble
- **机制**: MaskCLIP dense 的价值在 per-patch 2D map; 对任意形状 tile 取均值 = 聚合粒度不变 (仍 1 tile 1 向量) 但特征判别方差坍缩 4× (pos_sim std 0.010 vs 0.037)——均值化丢掉的恰是它相对 global embedding 的全部优势
- **诊断方法**: 注入/替换特征前先测文本相似度的 tile 间 std 与 relev 动态范围 (p95-p5), 坍缩特征免跑全链条 (8.17 方法的推广)
- **反直觉发现**: 3D 渲染特征 (AE+多视角+per-pixel) 比分段常数 tile 图高 +23pp——"2D tile 级上限" 对 3D 管线不是上限而是下限参考, 评估 2D 特征替换价值时不能只看 tile map 对比
- **快速 smooth 复刻**: eval/utils.smooth 的 7×7 majority 逐像素循环 (~1.3s/mask) 可用积分图精确复刻 (含 min(i+s, h-1) 独占上界的边界行为与 tie→0), 0.012s/mask, 已逐像素验证一致 (eval/dense_upper_test.py)

### 8.19 判别性快测必须同时看方差与空间对齐 (EXP-026)
- **教训**: EXP-025 用 pos_sim std 判死 tile-mean (0.010, 坍缩) 是对的; 但 EXP-026 per-patch std=0.0266 (不坍缩) 依然全灭——方差只证明"特征有差异", 不证明"差异是物体相关的"
- **完整判据**: 判别性快测需要三件套: ① tile/patch 级 pos_sim std (方差) ② 大物体 GT 内外 relev 差 (rel_in vs rel_out, 空间对齐) ③ 协议内 IoU。三者全过才值得跑管线
- **背景知识**: CLIP patch 特征与 text 的余弦相似度没有空间 grounding——CLIP 的对比训练只对齐 global image-text, 这是模型性质不是聚合协议性质; 换任何 CLIP 变体 (L/14, SigLIP, EVA) 的 dense patch 都会是同样结果 (EXP-022/025/026 三角互证)
- **滑窗 dense map 快测模板**: 224 crop + stride 112 + patch 块赋值 + 重叠均值 + 再归一化, 每 16px 一特征, eval/dense_patch_upper_test.py 可直接复用于任何教师候选

### 8.20 图像端与 text 端改进不可叠加 (EXP-027)
- **发现**: tile 编码填充方案 (blur/black) 在无模板 eval 下 +2~4pp, 但 +7模板 ensemble 后全部低于原特征 (-3~-7pp)——模板 ensemble 的收益依赖黑背景特征的"极化"分布
- **方法论**: 评估任何图像端编码改进必须同时跑 ±templates, 只看无模板数字会误判方向 (SOTA 协议带 ensemble)
- **快测基建**: _s.npy 可直接反推每 tile 像素 mask (s[li]==t), 免重跑 SAM 即可做任何 tile 级编码方案的免训练快测 (eval/masked_pool_test.py)
- **死代码炸弹**: eval/openclip_encoder.py 曾有 encode_image(mask=None) 无条件转发给不支持的底层 open_clip——无调用方所以长期潜伏, 新脚本一旦调用即 TypeError; 修复已做, 引以为戒: 改公共封装要 grep 全部调用方

## 8.21 EXP-031/032 流水线三坑 (2026-09-29)
- AE ckpt 保存门槛: autoencoder/train.py 仅在 epoch>95 时评估并保存 best_ckpt; --num_epochs 20 会导致训练正常完成但永远不落盘 (best_loss 显示初始值 100 假象, 误判为发散)。AE 必须用默认 100 epochs
- AE lr: 历史全部用默认 1e-4; 传 0.001 会真发散 (loss NaN)
- train.py 的 -m 约定: -m 只传不含 level 的基名 (如 output/teatime_xxx), train.py L226 自动追加 _<feature_level>; 若 -m 已含 level (output/teatime_xxx_1) 产物会落 _1_1, 后续 render/eval 路径全断
- 流水线防呆: 每步产物校验 (ls ckpt/render 计数) + set -e 是正确设计, 三次失败都在校验点被截停, 未污染下游
