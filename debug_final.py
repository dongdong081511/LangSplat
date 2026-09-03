"""深度诊断相似度和特征"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip
import cv2
from scipy import ndimage

class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)

    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

device = torch.device('cuda')

# 加载模型
encoder_dims = [256, 128, 64, 32, 3]
decoder_dims = [16, 32, 64, 128, 256, 256, 512]
ae = Autoencoder(encoder_dims, decoder_dims)
ckpt = torch.load('autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth', map_location='cpu')
ae.load_state_dict(ckpt)
ae = ae.to(device)
ae.eval()

model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16')

print("=== 问题诊断 ===")

# 1. 检查渲染特征与GT language feature的对比
print("\n[1] 对比渲染特征与GT language features")

# 加载GT language feature
gt_feat_path = 'output/sofa_data/language_features/02.npy'  # 对应segmentation目录02
if os.path.exists(gt_feat_path):
    gt_feat = np.load(gt_feat_path)
    print(f"GT language feature: {gt_feat_path}")
    print(f"  形状: {gt_feat.shape}")
    print(f"  范围: [{gt_feat.min():.4f}, {gt_feat.max():.4f}]")
else:
    print(f"GT feature不存在: {gt_feat_path}")

# 加载渲染特征
render_feat_path = 'output/sofa_1/train/ours_None/renders_npy/00002.npy'
render_feat = np.load(render_feat_path)
print(f"\n渲染特征: {render_feat_path}")
print(f"  形状: {render_feat.shape}")
print(f"  范围: [{render_feat.min():.4f}, {render_feat.max():.4f}]")

# 解码渲染特征
feat_flat = render_feat.reshape(-1, 3)
feat_tensor = torch.from_numpy(feat_flat).float().to(device)
with torch.no_grad():
    decoded = ae.decode(feat_tensor).cpu().numpy()
decoded = decoded.reshape(render_feat.shape[0], render_feat.shape[1], 512)
decoded_norm = decoded / (np.linalg.norm(decoded, axis=-1, keepdims=True) + 1e-8)

print(f"解码后特征:")
print(f"  形状: {decoded.shape}")

# 2. 检查GT language feature是否是512维
if os.path.exists(gt_feat_path):
    if len(gt_feat.shape) == 3 and gt_feat.shape[2] == 512:
        # 直接使用GT feature
        gt_feat_norm = gt_feat / (np.linalg.norm(gt_feat, axis=-1, keepdims=True) + 1e-8)
        
        # 计算GT特征和渲染特征的相似度
        # 这可以判断渲染质量
        diff = np.abs(gt_feat_norm - decoded_norm).mean()
        cos_sim = (gt_feat_norm * decoded_norm).sum(axis=-1).mean()
        print(f"\nGT vs 渲染特征:")
        print(f"  L1差异: {diff:.4f}")
        print(f"  余弦相似度: {cos_sim:.4f}")

# 3. 检查CLIP编码的text feature
print("\n[2] CLIP text features")
categories = ['Xbox wireless controller', 'a stack of UNO cards', 
              'a red Nintendo Switch joy-con controller', 'Pikachu', 'Gundam']

for cat in categories:
    text_tokens = tokenizer([cat]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().numpy()[0]
    
    # 计算与渲染特征的相似度
    similarity = np.dot(decoded_norm, text_feat)
    print(f"\n'{cat}':")
    print(f"  相似度: min={similarity.min():.4f}, max={similarity.max():.4f}, mean={similarity.mean():.4f}")
    print(f"  >0.2: {(similarity > 0.2).sum()} pixels ({(similarity > 0.2).mean()*100:.2f}%)")
    print(f"  >0.15: {(similarity > 0.15).sum()} pixels ({(similarity > 0.15).mean()*100:.2f}%)")
    print(f"  >0.1: {(similarity > 0.1).sum()} pixels ({(similarity > 0.1).mean()*100:.2f}%)")

# 4. 检查GT feature本身的分布
print("\n[3] GT language feature分布")
if os.path.exists(gt_feat_path) and len(gt_feat.shape) == 3 and gt_feat.shape[2] == 512:
    gt_feat_norm = gt_feat / (np.linalg.norm(gt_feat, axis=-1, keepdims=True) + 1e-8)
    for cat in categories:
        text_tokens = tokenizer([cat]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu().numpy()[0]
        
        similarity = np.dot(gt_feat_norm, text_feat)
        print(f"\n'{cat}' (GT feature):")
        print(f"  相似度: min={similarity.min():.4f}, max={similarity.max():.4f}, mean={similarity.mean():.4f}")
        print(f"  >0.3: {(similarity > 0.3).sum()} pixels ({(similarity > 0.3).mean()*100:.2f}%)")

# 5. 可视化最高相似度的位置
print("\n[4] 可视化分析")
for cat in ['Gundam', 'Pikachu']:
    text_tokens = tokenizer([cat]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().numpy()[0]
    
    similarity = np.dot(decoded_norm, text_feat)
    
    # 找到最高相似度的位置
    max_idx = np.unravel_index(np.argmax(similarity), similarity.shape)
    print(f"\n'{cat}':")
    print(f"  最高相似度位置: {max_idx}, 值: {similarity[max_idx]:.4f}")
    
    # 使用Otsu或自适应阈值
    from skimage import filters
    thresh = filters.threshold_otsu(similarity)
    print(f"  Otsu阈值: {thresh:.4f}")
    pred_mask = (similarity > thresh).astype(np.uint8)
    print(f"  Otsu预测像素: {pred_mask.sum()}")

# 6. 检查帧映射是否正确
print("\n[5] 检查帧映射")
print("segmentation目录 02 应该对应:")
print("  - 原始数据集图片 02.jpg")
print("  - sofa_data图片索引 2 -> 00002.npy")
print("\n原始数据集图片:")
orig_imgs = sorted(os.listdir('dataset/3D Open-vocabulary Segmentation datasets/sofa/images/'))
print(f"  {orig_imgs}")

# 检查02.jpg是否在原始数据集中
if '02.jpg' in orig_imgs:
    print("\n02.jpg存在于原始数据集!")
    # 检查sofa_data中02.jpg对应的索引
    sofa_imgs = sorted(os.listdir('output/sofa_data/images/'))
    idx = sofa_imgs.index('02.jpg')
    print(f"sofa_data中02.jpg索引: {idx} -> 渲染文件 {idx:05d}.npy")
