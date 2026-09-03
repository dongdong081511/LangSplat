#!/usr/bin/env python3
"""
诊断评估流程
"""
import numpy as np
import torch
import sys
import cv2
import json
sys.path.insert(0, '/home/xiedexia/project/LangSplat/eval')

from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork

device = 'cuda'

# 加载Sofa AE
ae_path = '/home/xiedexia/project/LangSplat/autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth'
ckpt = torch.load(ae_path, map_location=device)
ae = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to(device)
ae.load_state_dict(ckpt)
ae.eval()
print("AE loaded")

# 初始化CLIP模型
clip_model = OpenCLIPNetwork(device)
print("CLIP model loaded")

# 加载渲染特征（3个level）
renders = []
for level in [1, 2, 3]:
    path = f'/home/xiedexia/project/LangSplat/output/lerf_teatime_{level}/train/ours_None/renders_npy/00002.npy'
    renders.append(np.load(path))
print(f"Loaded {len(renders)} level renders, shape: {renders[0].shape}")

# 加载GT信息
gt_path = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs/label/teatime/frame_00002.json'
with open(gt_path) as f:
    gt_data = json.load(f)

labels = [obj['category'] for obj in gt_data['objects']]
print(f"GT labels: {labels}")

# 设置CLIP positives
clip_model.set_positives(labels)

# 解码渲染特征
sem_feats = []
for i, render in enumerate(renders):
    h, w, c = render.shape
    with torch.no_grad():
        render_tensor = torch.from_numpy(render).float().to(device)
        decoded = ae.decode(render_tensor.view(-1, 3))
        decoded = decoded.view(h, w, -1)
    sem_feats.append(decoded)
    
# Stack to (3, H, W, 512)
sem_feats = torch.stack(sem_feats)
print(f"Sem feats shape: {sem_feats.shape}")

# 计算relevancy map
n_levels, h, w, _ = sem_feats.shape
n_phrases = len(labels)

for j, label in enumerate(labels):
    print(f"\n=== Label: {label} ===")
    for i in range(n_levels):
        # 获取该level的特征
        feat = sem_feats[i].view(-1, 512)  # (H*W, 512)
        
        # 计算relevancy
        probs = clip_model.get_relevancy(feat, j)
        pos_prob = probs[:, 0].view(h, w)
        
        pct = 100.0 * (pos_prob > 0.5).float().mean().item()
        print(f"  Level {i}: pos_prob mean={pos_prob.mean().item():.4f}, max={pos_prob.max().item():.4f}, >0.5={pct:.1f}%")

# 检查CLIP text embedding和decoded feature的相似度
print("\n=== CLIP text embed vs decoded feature ===")
text_embeds = clip_model.pos_embeds
for i, label in enumerate(labels):
    text_embed = text_embeds[i]
    
    for level in range(3):
        feat = sem_feats[level].view(-1, 512)
        # 计算cosine similarity (feat已归一化，text_embed也已归一化)
        cos_sim = torch.mm(feat, text_embed.unsqueeze(1)).squeeze()
        print(f"  {label} Level {level}: cos_sim mean={cos_sim.mean().item():.4f}, max={cos_sim.max().item():.4f}, min={cos_sim.min().item():.4f}")

# 检查与原始CLIP特征的对比
print("\n=== 原始CLIP特征 vs 解码特征 ===")
original_feat = np.load('/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime/language_features/frame_00002_f.npy')
dim3_gt = np.load('/home/xiedexia/project/LangSplat/dataset/lerf_ovs/teatime/language_features_dim3/frame_00002_f.npy')

# 解码GT dim3
with torch.no_grad():
    decoded_gt = ae.decode(torch.from_numpy(dim3_gt).float().to(device)).cpu().numpy()

# 计算相似度
original_norm = original_feat / (np.linalg.norm(original_feat, axis=1, keepdims=True) + 1e-8)
decoded_gt_norm = decoded_gt / (np.linalg.norm(decoded_gt, axis=1, keepdims=True) + 1e-8)
cos_sims = np.sum(original_norm * decoded_gt_norm, axis=1)
print(f"GT dim3解码 vs 原始CLIP: cosine mean={cos_sims.mean():.4f}, >0.9={100*(cos_sims>0.9).mean():.1f}%")

# 用原始CLIP特征计算relevancy
print("\n=== 使用原始CLIP特征计算relevancy ===")
with torch.no_grad():
    original_tensor = torch.from_numpy(original_feat).float().to(device)
    for j, label in enumerate(labels):
        probs = clip_model.get_relevancy(original_tensor, j)
        pos_prob = probs[:, 0]
        pct = 100.0 * (pos_prob > 0.5).float().mean().item()
        print(f"  {label}: pos_prob mean={pos_prob.mean().item():.4f}, max={pos_prob.max().item():.4f}, >0.5={pct:.1f}%")
