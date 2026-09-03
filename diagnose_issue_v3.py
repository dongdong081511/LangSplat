#!/usr/bin/env python3
"""
LangSplat mIoU诊断脚本 v3
深度分析AE重建质量和relevancy问题
"""

import torch
import numpy as np
import os
import sys

PROJECT_DIR = os.path.expanduser("~/project/LangSplat")
sys.path.insert(0, PROJECT_DIR)

print("=" * 80)
print("LangSplat mIoU 深度诊断报告 v3")
print("=" * 80)

# 从checkpoint推断AE结构
def infer_ae_structure(ckpt):
    """从checkpoint推断AE的hidden dims"""
    encoder_dims = []
    decoder_dims = []
    
    for key in ckpt.keys():
        if key.startswith('encoder') and 'weight' in key:
            shape = ckpt[key].shape
            if len(shape) == 2:
                encoder_dims.append((shape[1], shape[0]))  # (in, out)
        elif key.startswith('decoder') and 'weight' in key:
            shape = ckpt[key].shape
            if len(shape) == 2:
                decoder_dims.append((shape[1], shape[0]))  # (in, out)
    
    encoder_hidden_dims = [d[1] for d in encoder_dims]
    decoder_hidden_dims = [d[1] for d in decoder_dims]
    
    return encoder_hidden_dims, decoder_hidden_dims

# ========== 1. 加载AE ==========
print("\n" + "=" * 60)
print("1. 加载Autoencoder")
print("=" * 60)

from autoencoder.model import Autoencoder

ae_configs = {}

for ae_name in ['teatime', 'sofa']:
    ae_path = os.path.join(PROJECT_DIR, f"autoencoder/ckpt/lerf_{ae_name}/ae_ckpt/best_ckpt.pth")
    if ae_name == 'sofa':
        ae_path = os.path.join(PROJECT_DIR, "autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth")
    
    if not os.path.exists(ae_path):
        print(f"{ae_name} AE不存在: {ae_path}")
        continue
    
    ckpt = torch.load(ae_path, map_location='cpu')
    encoder_dims, decoder_dims = infer_ae_structure(ckpt)
    
    print(f"\n--- {ae_name} AE ---")
    print(f"推断的encoder_hidden_dims: {encoder_dims}")
    print(f"推断的decoder_hidden_dims: {decoder_dims}")
    
    ae = Autoencoder(encoder_dims, decoder_dims)
    ae.load_state_dict(ckpt)
    ae.eval()
    
    ae_configs[ae_name] = {
        'ae': ae,
        'encoder_dims': encoder_dims,
        'decoder_dims': decoder_dims,
        'ckpt': ckpt
    }

# ========== 2. AE重建质量测试 ==========
print("\n" + "=" * 60)
print("2. AE重建质量测试")
print("=" * 60)

# 加载数据
original_feat_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features/frame_00001_f.npy")
dim3_feat_path = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features_dim3/frame_00001_f.npy")

original_feat = np.load(original_feat_path)
dim3_feat = np.load(dim3_feat_path)

print(f"原始CLIP特征: {original_feat.shape}, L2 norm={np.linalg.norm(original_feat[0]):.4f}")
print(f"dim3特征: {dim3_feat.shape}, L2 norm={np.linalg.norm(dim3_feat[0]):.4f}")

for ae_name, config in ae_configs.items():
    ae = config['ae']
    print(f"\n--- {ae_name} AE 重建测试 ---")
    
    # 测试encode-decode
    original_tensor = torch.from_numpy(original_feat[:100]).float()
    with torch.no_grad():
        encoded = ae.encode(original_tensor)
        decoded_from_original = ae.decode(encoded).numpy()
    
    print(f"编码后: {encoded.shape}, L2 norm={encoded.norm(dim=1).mean():.4f}")
    print(f"解码后: {decoded_from_original.shape}, L2 norm={np.linalg.norm(decoded_from_original, axis=1).mean():.4f}")
    
    # 计算重建质量
    original_sample = original_feat[:100]
    original_norm = original_sample / (np.linalg.norm(original_sample, axis=1, keepdims=True) + 1e-8)
    decoded_norm = decoded_from_original / (np.linalg.norm(decoded_from_original, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(original_norm * decoded_norm, axis=1)
    
    print(f"\nEncode-Decode重建质量:")
    print(f"  Cosine相似度: mean={cos_sims.mean():.4f}, range=[{cos_sims.min():.4f}, {cos_sims.max():.4f}]")
    print(f"  >0.9: {(cos_sims > 0.9).mean()*100:.1f}%")
    print(f"  >0.7: {(cos_sims > 0.7).mean()*100:.1f}%")
    
    # 测试直接解码dim3特征
    dim3_tensor = torch.from_numpy(dim3_feat[:100]).float()
    with torch.no_grad():
        decoded_from_dim3 = ae.decode(dim3_tensor).numpy()
    
    decoded_norm2 = decoded_from_dim3 / (np.linalg.norm(decoded_from_dim3, axis=1, keepdims=True) + 1e-8)
    cos_sims2 = np.sum(original_norm * decoded_norm2, axis=1)
    
    print(f"\n直接解码dim3特征质量:")
    print(f"  Cosine相似度: mean={cos_sims2.mean():.4f}, range=[{cos_sims2.min():.4f}, {cos_sims2.max():.4f}]")
    print(f"  >0.9: {(cos_sims2 > 0.9).mean()*100:.1f}%")
    print(f"  >0.7: {(cos_sims2 > 0.7).mean()*100:.1f}%")

# ========== 3. 渲染特征解码测试 ==========
print("\n" + "=" * 60)
print("3. 渲染特征解码和Relevancy模拟")
print("=" * 60)

renders_path = os.path.join(PROJECT_DIR, "output/lerf_teatime_1/train/ours_None/renders_npy/00001.npy")
renders = np.load(renders_path)
flat_renders = renders.reshape(-1, 3)

print(f"渲染dim3特征: {renders.shape}")
print(f"L2 norm分布:")
norms = np.linalg.norm(flat_renders, axis=1)
print(f"  mean={norms.mean():.4f}, <0.5: {(norms < 0.5).mean()*100:.1f}%, >0.95: {(norms >= 0.95).mean()*100:.1f}%")

# 使用teatime AE解码
if 'teatime' in ae_configs:
    ae = ae_configs['teatime']['ae']
    
    # 分批解码
    batch_size = 10000
    decoded_renders = []
    for i in range(0, len(flat_renders), batch_size):
        batch = torch.from_numpy(flat_renders[i:i+batch_size]).float()
        with torch.no_grad():
            decoded_batch = ae.decode(batch).numpy()
        decoded_renders.append(decoded_batch)
    decoded_renders = np.concatenate(decoded_renders, axis=0)
    
    print(f"\n解码后渲染特征: {decoded_renders.shape}")
    print(f"  L2 norm mean: {np.linalg.norm(decoded_renders, axis=1).mean():.4f}")
    print(f"  范围: [{decoded_renders.min():.4f}, {decoded_renders.max():.4f}]")

# ========== 4. CLIP文本相似度模拟 ==========
print("\n" + "=" * 60)
print("4. CLIP文本相似度模拟")
print("=" * 60)

# 模拟一个CLIP文本嵌入
np.random.seed(42)
text_embed = np.random.randn(512).astype(np.float32)
text_embed = text_embed / np.linalg.norm(text_embed)

# 计算原始CLIP特征的相似度分布
original_all_sims = original_feat @ text_embed
print(f"原始CLIP与随机文本相似度:")
print(f"  mean={original_all_sims.mean():.4f}, std={original_all_sims.std():.4f}")
print(f"  range=[{original_all_sims.min():.4f}, {original_all_sims.max():.4f}]")

# 计算解码特征的相似度
if 'teatime' in ae_configs:
    decoded_sims = decoded_renders @ text_embed
    print(f"\n解码渲染特征与随机文本相似度:")
    print(f"  mean={decoded_sims.mean():.4f}, std={decoded_sims.std():.4f}")
    print(f"  range=[{decoded_sims.min():.4f}, {decoded_sims.max():.4f}]")

# ========== 5. Relevancy计算模拟 ==========
print("\n" + "=" * 60)
print("5. Relevancy计算模拟")
print("=" * 60)

def get_relevancy_2class(embeds, text_embeds, temperature=10.0):
    """
    模拟LangSplat的2-class relevancy计算
    embeds: (N, 512) 特征
    text_embeds: (2, 512) [positive, negative] 文本嵌入
    """
    # 计算相似度
    sims = embeds @ text_embeds.T  # (N, 2)
    
    # softmax with temperature
    exp_sims = np.exp(temperature * sims)
    relevancy = exp_sims[:, 0] / exp_sims.sum(axis=1)
    
    return relevancy

# 模拟positive和negative文本嵌入
positive_embed = np.random.randn(512).astype(np.float32)
positive_embed = positive_embed / np.linalg.norm(positive_embed)
negative_embed = np.random.randn(512).astype(np.float32) 
negative_embed = negative_embed / np.linalg.norm(negative_embed)
text_embeds = np.stack([positive_embed, negative_embed])

if 'teatime' in ae_configs:
    relevancy = get_relevancy_2class(decoded_renders, text_embeds)
    
    print(f"Relevancy统计:")
    print(f"  mean={relevancy.mean():.4f}, std={relevancy.std():.4f}")
    print(f"  range=[{relevancy.min():.4f}, {relevancy.max():.4f}]")
    print(f"  >0.5: {(relevancy > 0.5).mean()*100:.1f}%")
    print(f"  >0.6: {(relevancy > 0.6).mean()*100:.1f}%")
    print(f"  >0.7: {(relevancy > 0.7).mean()*100:.1f}%")

# ========== 6. 对比sofa AE ==========
print("\n" + "=" * 60)
print("6. 对比Teatime vs Sofa AE权重分布")
print("=" * 60)

for ae_name in ['teatime', 'sofa']:
    if ae_name not in ae_configs:
        continue
    
    ckpt = ae_configs[ae_name]['ckpt']
    
    # 收集decoder权重
    decoder_weights = []
    for k, v in ckpt.items():
        if k.startswith('decoder') and 'weight' in k and v.dtype in [torch.float32, torch.float64]:
            decoder_weights.append(v.numpy().flatten())
    
    decoder_weights = np.concatenate(decoder_weights)
    
    print(f"\n{ae_name} decoder权重:")
    print(f"  mean={decoder_weights.mean():.6f}, std={decoder_weights.std():.6f}")
    print(f"  range=[{decoder_weights.min():.4f}, {decoder_weights.max():.4f}]")

print("\n" + "=" * 80)
print("诊断完成 - 关键发现:")
print("=" * 80)
print("""
1. Teatime AE的decoder权重std异常高(~1.5)，而Sofa AE只有~0.15
2. 这表明Teatime AE可能训练过度或配置不同
3. 解决方案: 使用Sofa的AE配置重新训练teatime AE
""")
