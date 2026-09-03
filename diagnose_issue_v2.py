#!/usr/bin/env python3
"""
LangSplat mIoU诊断脚本 v2
深入分析AE重建质量
"""

import torch
import numpy as np
import os
import sys

PROJECT_DIR = os.path.expanduser("~/project/LangSplat")
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("LangSplat mIoU 深度诊断报告 v2")
print("=" * 80)

# ========== 1. 查看Autoencoder的正确API ==========
print("\n" + "=" * 60)
print("1. 检查Autoencoder结构")
print("=" * 60)

try:
    from autoencoder.model import Autoencoder
    import inspect
    print(f"Autoencoder __init__ 签名: {inspect.signature(Autoencoder.__init__)}")
    print(f"\nAutoencoder 类定义:")
    print(inspect.getsource(Autoencoder))
except Exception as e:
    print(f"获取Autoencoder信息失败: {e}")

# ========== 2. 直接加载并测试AE ==========
print("\n" + "=" * 60)
print("2. AE重建质量测试")
print("=" * 60)

# 加载原始CLIP特征
original_feat_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features/frame_00001_f.npy")
dim3_feat_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features_dim3/frame_00001_f.npy")

original_feat = np.load(original_feat_path)
dim3_feat = np.load(dim3_feat_path)

print(f"原始CLIP特征: {original_feat.shape}, L2 norm={np.linalg.norm(original_feat[0]):.4f}")
print(f"dim3特征: {dim3_feat.shape}, L2 norm={np.linalg.norm(dim3_feat[0]):.4f}")

# 尝试不同的AE加载方式
ae_path_teatime = os.path.join(PROJECT_DIR, "autoencoder/ckpt/lerf_teatime/ae_ckpt/best_ckpt.pth")
ae_path_sofa = os.path.join(PROJECT_DIR, "autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth")

for ae_name, ae_path in [('teatime', ae_path_teatime), ('sofa', ae_path_sofa)]:
    if not os.path.exists(ae_path):
        print(f"\n{ae_name} AE不存在: {ae_path}")
        continue
        
    print(f"\n--- {ae_name} AE ---")
    ckpt = torch.load(ae_path, map_location='cpu')
    
    # 创建AE模型
    from autoencoder.model import Autoencoder
    ae = Autoencoder()  # 使用默认参数
    ae.load_state_dict(ckpt)
    ae.eval()
    
    # 测试编码
    original_tensor = torch.from_numpy(original_feat[:100]).float()
    with torch.no_grad():
        encoded = ae.encode(original_tensor)
        decoded = ae.decode(encoded)
    
    encoded_np = encoded.numpy()
    decoded_np = decoded.numpy()
    
    print(f"编码后 (dim3): {encoded_np.shape}")
    print(f"  范围: [{encoded_np.min():.4f}, {encoded_np.max():.4f}]")
    print(f"  L2 norm: {np.linalg.norm(encoded_np, axis=1).mean():.4f}")
    
    print(f"解码后 (512维): {decoded_np.shape}")
    print(f"  范围: [{decoded_np.min():.4f}, {decoded_np.max():.4f}]")
    print(f"  L2 norm: {np.linalg.norm(decoded_np, axis=1).mean():.4f}")
    
    # 计算重建质量
    original_sample = original_feat[:100]
    
    # 归一化后计算cosine相似度
    original_norm = original_sample / (np.linalg.norm(original_sample, axis=1, keepdims=True) + 1e-8)
    decoded_norm = decoded_np / (np.linalg.norm(decoded_np, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(original_norm * decoded_norm, axis=1)
    
    print(f"\n重建质量 (原始CLIP vs encode-decode):")
    print(f"  Cosine相似度: mean={cos_sims.mean():.4f}, range=[{cos_sims.min():.4f}, {cos_sims.max():.4f}]")
    print(f"  >0.9: {(cos_sims > 0.9).mean()*100:.1f}%")
    print(f"  >0.7: {(cos_sims > 0.7).mean()*100:.1f}%")
    
    # 测试直接解码dim3特征
    dim3_tensor = torch.from_numpy(dim3_feat[:100]).float()
    with torch.no_grad():
        decoded_from_dim3 = ae.decode(dim3_tensor).numpy()
    
    decoded_norm2 = decoded_from_dim3 / (np.linalg.norm(decoded_from_dim3, axis=1, keepdims=True) + 1e-8)
    cos_sims2 = np.sum(original_norm * decoded_norm2, axis=1)
    
    print(f"\n解码质量 (dim3特征解码 vs 原始CLIP):")
    print(f"  Cosine相似度: mean={cos_sims2.mean():.4f}, range=[{cos_sims2.min():.4f}, {cos_sims2.max():.4f}]")
    print(f"  >0.9: {(cos_sims2 > 0.9).mean()*100:.1f}%")
    print(f"  >0.7: {(cos_sims2 > 0.7).mean()*100:.1f}%")

# ========== 3. 检查AE训练配置 ==========
print("\n" + "=" * 60)
print("3. AE训练配置检查")
print("=" * 60)

ae_dirs = [
    os.path.join(PROJECT_DIR, "autoencoder/ckpt/lerf_teatime"),
    os.path.join(PROJECT_DIR, "autoencoder/ckpt/sofa"),
]

for ae_dir in ae_dirs:
    if os.path.exists(ae_dir):
        print(f"\n{ae_dir}:")
        for f in os.listdir(ae_dir):
            fpath = os.path.join(ae_dir, f)
            if os.path.isfile(fpath):
                print(f"  {f}: {os.path.getsize(fpath)} bytes")

# ========== 4. 计算CLIP文本相似度模拟 ==========
print("\n" + "=" * 60)
print("4. CLIP文本相似度模拟")
print("=" * 60)

# 模拟CLIP文本嵌入 (随机单位向量作为示例)
np.random.seed(42)
text_embed = np.random.randn(512).astype(np.float32)
text_embed = text_embed / np.linalg.norm(text_embed)

# 计算原始CLIP特征与文本的相似度
original_sims = original_feat @ text_embed
print(f"原始CLIP特征与随机文本的相似度:")
print(f"  mean={original_sims.mean():.4f}, std={original_sims.std():.4f}")
print(f"  range=[{original_sims.min():.4f}, {original_sims.max():.4f}]")

# 加载teatime AE并解码
ckpt = torch.load(ae_path_teatime, map_location='cpu')
from autoencoder.model import Autoencoder
ae = Autoencoder()
ae.load_state_dict(ckpt)
ae.eval()

dim3_tensor = torch.from_numpy(dim3_feat).float()
with torch.no_grad():
    decoded_feat = ae.decode(dim3_tensor).numpy()

# 计算解码特征与文本的相似度
decoded_sims = decoded_feat @ text_embed
print(f"\n解码特征与随机文本的相似度:")
print(f"  mean={decoded_sims.mean():.4f}, std={decoded_sims.std():.4f}")
print(f"  range=[{decoded_sims.min():.4f}, {decoded_sims.max():.4f}]")

# ========== 5. 检查渲染特征的分布 ==========
print("\n" + "=" * 60)
print("5. 渲染特征分布深度分析")
print("=" * 60)

renders_path = os.path.join(PROJECT_DIR, "output/lerf_teatime_1/train/ours_None/renders_npy/00001.npy")
renders = np.load(renders_path)
flat_renders = renders.reshape(-1, 3)

print(f"Shape: {renders.shape}")
print(f"总像素数: {flat_renders.shape[0]}")

# L2 norm分布
norms = np.linalg.norm(flat_renders, axis=1)
print(f"\nL2 norm分布:")
print(f"  <0.5: {(norms < 0.5).mean()*100:.1f}%")
print(f"  0.5-0.8: {((norms >= 0.5) & (norms < 0.8)).mean()*100:.1f}%")
print(f"  0.8-0.95: {((norms >= 0.8) & (norms < 0.95)).mean()*100:.1f}%")
print(f"  >0.95: {(norms >= 0.95).mean()*100:.1f}%")

# 解码渲染特征
renders_tensor = torch.from_numpy(flat_renders).float()
with torch.no_grad():
    # 分批处理避免OOM
    batch_size = 10000
    decoded_renders = []
    for i in range(0, len(renders_tensor), batch_size):
        batch = renders_tensor[i:i+batch_size]
        decoded_batch = ae.decode(batch).numpy()
        decoded_renders.append(decoded_batch)
    decoded_renders = np.concatenate(decoded_renders, axis=0)

print(f"\n解码后渲染特征:")
print(f"  Shape: {decoded_renders.shape}")
print(f"  L2 norm: mean={np.linalg.norm(decoded_renders, axis=1).mean():.4f}")

# 计算与文本的相似度
renders_sims = decoded_renders @ text_embed
print(f"\n解码渲染特征与随机文本的相似度:")
print(f"  mean={renders_sims.mean():.4f}, std={renders_sims.std():.4f}")
print(f"  range=[{renders_sims.min():.4f}, {renders_sims.max():.4f}]")

# 模拟relevancy计算
def get_relevancy(sims, temperature=10.0):
    """模拟LangSplat的relevancy计算"""
    exp_sims = np.exp(temperature * sims)
    relevancy = exp_sims / (exp_sims + 1)
    return relevancy

relevancy = get_relevancy(renders_sims)
print(f"\nRelevancy值 (模拟):")
print(f"  mean={relevancy.mean():.4f}")
print(f"  range=[{relevancy.min():.4f}, {relevancy.max():.4f}]")
print(f"  >0.5: {(relevancy > 0.5).mean()*100:.1f}%")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
