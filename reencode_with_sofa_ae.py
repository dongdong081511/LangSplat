#!/usr/bin/env python3
"""
使用Sofa AE重新编码teatime数据
"""

import torch
import numpy as np
import os
import sys
from tqdm import tqdm

PROJECT_DIR = os.path.expanduser("~/project/LangSplat")
sys.path.insert(0, PROJECT_DIR)

from autoencoder.model import Autoencoder

# 加载Sofa AE
ae_path = os.path.join(PROJECT_DIR, "autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth")
ckpt = torch.load(ae_path, map_location='cpu')

# 推断AE结构
encoder_dims = [256, 128, 64, 32, 3]
decoder_dims = [16, 32, 64, 128, 256, 256, 512]

ae = Autoencoder(encoder_dims, decoder_dims)
ae.load_state_dict(ckpt)
ae.eval()
print("Sofa AE loaded successfully")

# 源数据和目标路径
src_dir = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features")
dst_dir = os.path.join(PROJECT_DIR, "dataset/lerf_ovs/teatime/language_features_dim3_sofa")
os.makedirs(dst_dir, exist_ok=True)

# 获取所有特征文件
feat_files = sorted([f for f in os.listdir(src_dir) if f.endswith('_f.npy')])

print(f"Processing {len(feat_files)} files...")

for feat_file in tqdm(feat_files):
    # 加载原始CLIP特征
    feat_path = os.path.join(src_dir, feat_file)
    feat = np.load(feat_path)
    
    # 编码为dim3
    with torch.no_grad():
        feat_tensor = torch.from_numpy(feat).float()
        dim3_feat = ae.encode(feat_tensor).numpy()
    
    # 保存
    dst_path = os.path.join(dst_dir, feat_file)
    np.save(dst_path, dim3_feat)
    
    # 复制seg_map文件
    seg_file = feat_file.replace('_f.npy', '_s.npy')
    src_seg = os.path.join(src_dir, seg_file)
    dst_seg = os.path.join(dst_dir, seg_file)
    if os.path.exists(src_seg):
        import shutil
        shutil.copy(src_seg, dst_seg)

print(f"Done! Saved to {dst_dir}")

# 验证编码质量
print("\n验证编码质量:")
sample_feat = np.load(os.path.join(src_dir, feat_files[0]))
sample_dim3 = np.load(os.path.join(dst_dir, feat_files[0]))

with torch.no_grad():
    decoded = ae.decode(torch.from_numpy(sample_dim3).float()).numpy()

# 计算cosine相似度
original_norm = sample_feat / (np.linalg.norm(sample_feat, axis=1, keepdims=True) + 1e-8)
decoded_norm = decoded / (np.linalg.norm(decoded, axis=1, keepdims=True) + 1e-8)
cos_sims = np.sum(original_norm * decoded_norm, axis=1)

print(f"Cosine相似度: mean={cos_sims.mean():.4f}, >0.9: {(cos_sims > 0.9).mean()*100:.1f}%")
