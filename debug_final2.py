"""深度诊断 - 使用正确的Autoencoder结构"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip
import cv2

# 使用正确的Autoencoder结构 (从eval_sofa.py复制)
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
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
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

# 加载模型 - 使用正确的dims
encoder_dims = [256, 128, 64, 32, 3]
decoder_dims = [16, 32, 64, 128, 256, 256, 512]
ae = Autoencoder(encoder_dims, decoder_dims)
ckpt = torch.load('autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth', map_location='cpu')
ae.load_state_dict(ckpt)
ae = ae.to(device)
ae.eval()
print("Autoencoder加载成功")

model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16')
print("CLIP加载成功")

print("\n" + "="*60)
print("问题诊断")
print("="*60)

# 1. 检查GT language feature
print("\n[1] GT language features分析")
gt_feat_dir = 'output/sofa_data/language_features'
gt_files = sorted(glob.glob(f'{gt_feat_dir}/*.npy'))
print(f"GT feature文件数: {len(gt_files)}")

# 加载一个GT feature
gt_feat_path = 'output/sofa_data/language_features/02.npy'
if os.path.exists(gt_feat_path):
    gt_feat = np.load(gt_feat_path)
    print(f"GT feature {gt_feat_path}:")
    print(f"  形状: {gt_feat.shape}")
    print(f"  dtype: {gt_feat.dtype}")
    print(f"  范围: [{gt_feat.min():.4f}, {gt_feat.max():.4f}]")
    
    # GT feature可能是512维，需要编码成3维才能和渲染特征比较
    if gt_feat.shape[-1] == 512:
        gt_feat_norm = gt_feat / (np.linalg.norm(gt_feat, axis=-1, keepdims=True) + 1e-8)
        
        # 用CLIP text编码计算GT feature的相似度
        print("\nGT feature + CLIP text 相似度:")
        categories = ['Gundam', 'Pikachu', 'Xbox wireless controller', 
                      'a stack of UNO cards', 'a red Nintendo Switch joy-con controller']
        for cat in categories:
            text_tokens = tokenizer([cat]).to(device)
            with torch.no_grad():
                text_feat = model.encode_text(text_tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu().numpy()[0]
            
            similarity = np.dot(gt_feat_norm, text_feat)
            print(f"  '{cat}': max={similarity.max():.4f}, mean={similarity.mean():.4f}, >0.3: {(similarity > 0.3).mean()*100:.1f}%")

# 2. 检查渲染特征
print("\n[2] 渲染特征分析")
render_feat_path = 'output/sofa_1/train/ours_None/renders_npy/00002.npy'
render_feat = np.load(render_feat_path)
print(f"渲染特征 {render_feat_path}:")
print(f"  形状: {render_feat.shape}")
print(f"  范围: [{render_feat.min():.4f}, {render_feat.max():.4f}]")

# 解码渲染特征
feat_flat = render_feat.reshape(-1, 3)
feat_tensor = torch.from_numpy(feat_flat).float().to(device)
with torch.no_grad():
    decoded = ae.decode(feat_tensor).cpu().numpy()
decoded = decoded.reshape(render_feat.shape[0], render_feat.shape[1], 512)
decoded_norm = decoded / (np.linalg.norm(decoded, axis=-1, keepdims=True) + 1e-8)
print(f"解码后:")
print(f"  形状: {decoded.shape}")
print(f"  范围: [{decoded.min():.4f}, {decoded.max():.4f}]")

# 3. 渲染特征 vs CLIP text
print("\n[3] 渲染特征(解码后) + CLIP text 相似度:")
for cat in ['Gundam', 'Pikachu', 'Xbox wireless controller']:
    text_tokens = tokenizer([cat]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().numpy()[0]
    
    similarity = np.dot(decoded_norm, text_feat)
    print(f"  '{cat}': max={similarity.max():.4f}, mean={similarity.mean():.4f}, >0.2: {(similarity > 0.2).mean()*100:.1f}%")

# 4. 对比GT和渲染特征
if os.path.exists(gt_feat_path) and gt_feat.shape[-1] == 512:
    print("\n[4] GT vs 渲染特征对比:")
    # 需要统一分辨率
    if gt_feat.shape[:2] != decoded.shape[:2]:
        print(f"  分辨率不同: GT={gt_feat.shape[:2]}, 渲染={decoded.shape[:2]}")
        # 缩放GT到渲染分辨率
        from scipy.ndimage import zoom
        scale_h = decoded.shape[0] / gt_feat.shape[0]
        scale_w = decoded.shape[1] / gt_feat.shape[1]
        gt_feat_scaled = zoom(gt_feat, (scale_h, scale_w, 1), order=1)
        gt_feat_scaled_norm = gt_feat_scaled / (np.linalg.norm(gt_feat_scaled, axis=-1, keepdims=True) + 1e-8)
    else:
        gt_feat_scaled_norm = gt_feat_norm
    
    # 计算余弦相似度
    cos_sim = (gt_feat_scaled_norm * decoded_norm).sum(axis=-1)
    print(f"  像素级余弦相似度: mean={cos_sim.mean():.4f}, max={cos_sim.max():.4f}")

# 5. 检查帧映射
print("\n[5] 帧映射验证:")
print("segmentation目录02 -> 对应原始图片02.jpg")
sofa_imgs = sorted(os.listdir('output/sofa_data/images/'))
if '02.jpg' in sofa_imgs:
    idx = sofa_imgs.index('02.jpg')
    print(f"sofa_data中02.jpg索引: {idx}")
    print(f"对应的渲染文件: {idx:05d}.npy")
    print(f"验证: 存在? {os.path.exists(f'output/sofa_1/train/ours_None/renders_npy/{idx:05d}.npy')}")
