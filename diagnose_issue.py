#!/usr/bin/env python3
"""
LangSplat mIoU诊断脚本
诊断relevancy值过低的问题
"""

import torch
import numpy as np
import os
import sys

# 项目路径
PROJECT_DIR = os.path.expanduser("~/project/LangSplat")
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("LangSplat mIoU 诊断报告")
print("=" * 80)

# ========== 1. 检查AE权重 ==========
print("\n" + "=" * 60)
print("1. AE Checkpoint 分析")
print("=" * 60)

ae_paths = {
    'teatime': os.path.join(PROJECT_DIR, "autoencoder/ckpt/lerf_teatime/ae_ckpt/best_ckpt.pth"),
    'sofa': os.path.join(PROJECT_DIR, "autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth"),
}

for name, ae_path in ae_paths.items():
    if os.path.exists(ae_path):
        print(f"\n--- {name} AE ---")
        ckpt = torch.load(ae_path, map_location='cpu')
        
        # 提取权重并分析
        all_weights = []
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor) and v.dtype in [torch.float32, torch.float64, torch.float16]:
                all_weights.append(v.flatten())
                print(f"  {k}: shape={v.shape}, mean={v.mean():.6f}, std={v.std():.6f}")
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            print(f"\n  [整体权重统计] mean={all_weights.mean():.6f}, std={all_weights.std():.6f}")
    else:
        print(f"\n--- {name} AE ---")
        print(f"  文件不存在: {ae_path}")

# ========== 2. 检查原始CLIP特征 ==========
print("\n" + "=" * 60)
print("2. 原始CLIP特征分析 (language_features)")
print("=" * 60)

original_feat_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features/frame_00001_f.npy")
if os.path.exists(original_feat_path):
    original_feat = np.load(original_feat_path)
    print(f"Shape: {original_feat.shape}")
    print(f"dtype: {original_feat.dtype}")
    print(f"范围: [{original_feat.min():.4f}, {original_feat.max():.4f}]")
    print(f"mean: {original_feat.mean():.4f}, std: {original_feat.std():.4f}")
    
    # 检查L2 norm
    norms = np.linalg.norm(original_feat, axis=1)
    print(f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, range=[{norms.min():.4f}, {norms.max():.4f}]")
else:
    print(f"文件不存在: {original_feat_path}")

# ========== 3. 检查dim3特征 ==========
print("\n" + "=" * 60)
print("3. dim3特征分析 (language_features_dim3)")
print("=" * 60)

dim3_feat_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features_dim3/frame_00001_f.npy")
if os.path.exists(dim3_feat_path):
    dim3_feat = np.load(dim3_feat_path)
    print(f"Shape: {dim3_feat.shape}")
    print(f"dtype: {dim3_feat.dtype}")
    print(f"范围: [{dim3_feat.min():.4f}, {dim3_feat.max():.4f}]")
    print(f"mean: {dim3_feat.mean():.4f}, std: {dim3_feat.std():.4f}")
    
    # 检查L2 norm
    norms = np.linalg.norm(dim3_feat, axis=1)
    print(f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, range=[{norms.min():.4f}, {norms.max():.4f}]")
else:
    print(f"文件不存在: {dim3_feat_path}")

# ========== 4. 检查渲染的dim3特征 ==========
print("\n" + "=" * 60)
print("4. 渲染的dim3特征分析 (renders_npy)")
print("=" * 60)

renders_path = os.path.join(PROJECT_DIR, "output/lerf_teatime_1/train/ours_None/renders_npy/00001.npy")
if not os.path.exists(renders_path):
    renders_path = os.path.join(PROJECT_DIR, "output/lerf_teatime_3/train/ours_None/renders_npy/00001.npy")

if os.path.exists(renders_path):
    renders = np.load(renders_path)
    print(f"文件: {renders_path}")
    print(f"Shape: {renders.shape}")
    print(f"dtype: {renders.dtype}")
    print(f"范围: [{renders.min():.4f}, {renders.max():.4f}]")
    print(f"mean: {renders.mean():.4f}, std: {renders.std():.4f}")
    
    # 检查L2 norm
    flat_renders = renders.reshape(-1, renders.shape[-1])
    norms = np.linalg.norm(flat_renders, axis=1)
    print(f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, range=[{norms.min():.4f}, {norms.max():.4f}]")
else:
    print(f"文件不存在")

# ========== 5. 加载AE并测试解码 ==========
print("\n" + "=" * 60)
print("5. AE解码测试")
print("=" * 60)

try:
    from autoencoder.model import Autoencoder
    
    # 加载teatime AE
    ae_path = os.path.join(PROJECT_DIR, "autoencoder/ckpt/lerf_teatime/ae_ckpt/best_ckpt.pth")
    if os.path.exists(ae_path):
        ckpt = torch.load(ae_path, map_location='cpu')
        
        ae = Autoencoder(feature_dim=512, latent_dim=3)
        ae.load_state_dict(ckpt)
        ae.eval()
        
        print(f"AE结构:\n{ae}")
        
        # 测试解码
        if os.path.exists(dim3_feat_path):
            dim3_tensor = torch.from_numpy(dim3_feat[:1000]).float()
            
            with torch.no_grad():
                decoded = ae.decode(dim3_tensor)
                decoded_np = decoded.numpy()
            
            print(f"\n解码后特征:")
            print(f"  Shape: {decoded_np.shape}")
            print(f"  范围: [{decoded_np.min():.4f}, {decoded_np.max():.4f}]")
            print(f"  mean: {decoded_np.mean():.4f}, std: {decoded_np.std():.4f}")
            
            # L2 norm
            norms = np.linalg.norm(decoded_np, axis=1)
            print(f"  L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}")
            
            # 计算与原始CLIP特征的cosine相似度
            if os.path.exists(original_feat_path):
                original_sample = original_feat[:1000]
                
                # 归一化
                original_norm = original_sample / (np.linalg.norm(original_sample, axis=1, keepdims=True) + 1e-8)
                decoded_norm = decoded_np / (np.linalg.norm(decoded_np, axis=1, keepdims=True) + 1e-8)
                
                # cosine相似度
                cos_sims = np.sum(original_norm * decoded_norm, axis=1)
                print(f"\n原始CLIP vs 解码特征 Cosine相似度:")
                print(f"  mean: {cos_sims.mean():.4f}, std: {cos_sims.std():.4f}")
                print(f"  range: [{cos_sims.min():.4f}, {cos_sims.max():.4f}]")
                print(f"  >0.9的比例: {(cos_sims > 0.9).mean()*100:.2f}%")
                print(f"  >0.7的比例: {(cos_sims > 0.7).mean()*100:.2f}%")
                print(f"  >0.5的比例: {(cos_sims > 0.5).mean()*100:.2f}%")

except Exception as e:
    print(f"AE加载/解码测试失败: {e}")
    import traceback
    traceback.print_exc()

# ========== 6. 检查seg_map ==========
print("\n" + "=" * 60)
print("6. seg_map分析")
print("=" * 60)

seg_map_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features/frame_00001_s.npy")
if os.path.exists(seg_map_path):
    seg_map = np.load(seg_map_path)
    print(f"Shape: {seg_map.shape}")
    print(f"dtype: {seg_map.dtype}")
    print(f"唯一值数量: {len(np.unique(seg_map))}")
    print(f"唯一值前20个: {np.unique(seg_map)[:20]}")

# ========== 7. 检查评估代码 ==========
print("\n" + "=" * 60)
print("7. 评估代码关键参数")
print("=" * 60)

eval_path = os.path.join(PROJECT_DIR, "eval/evaluate_iou_loc.py")
if os.path.exists(eval_path):
    with open(eval_path, 'r') as f:
        content = f.read()
    
    import re
    
    # 查找thresh
    thresh_matches = re.findall(r'thresh[^=]*=\s*([0-9.]+)', content)
    print(f"找到的thresh值: {thresh_matches}")
    
    # 查找get_relevancy函数
    if 'get_relevancy' in content:
        print("\n找到get_relevancy函数")
        func_match = re.search(r'def get_relevancy.*?(?=\ndef |\nclass |\Z)', content, re.DOTALL)
        if func_match:
            print(f"get_relevancy函数:\n{func_match.group(0)[:800]}...")

# ========== 8. 检查sofa数据对比 ==========
print("\n" + "=" * 60)
print("8. Sofa vs Teatime 数据对比")
print("=" * 60)

sofa_feat_path = os.path.join(PROJECT_DIR, "output/sofa_data/language_features/frame_00001_f.npy")
if os.path.exists(sofa_feat_path):
    sofa_feat = np.load(sofa_feat_path)
    print(f"Sofa原始CLIP特征:")
    print(f"  Shape: {sofa_feat.shape}")
    print(f"  L2 norm: mean={np.linalg.norm(sofa_feat, axis=1).mean():.4f}")
else:
    print("Sofa原始CLIP特征文件不存在")

sofa_dim3_path = os.path.join(PROJECT_DIR, "output/sofa_data/language_features_dim3/frame_00001_f.npy")
if os.path.exists(sofa_dim3_path):
    sofa_dim3 = np.load(sofa_dim3_path)
    print(f"\nSofa dim3特征:")
    print(f"  Shape: {sofa_dim3.shape}")
    print(f"  L2 norm: mean={np.linalg.norm(sofa_dim3, axis=1).mean():.4f}")
else:
    print("Sofa dim3特征文件不存在")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
