"""诊断GT feature - 正确的文件名格式"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip
from scipy.ndimage import zoom

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

    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

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

print("="*60)
print("GT feature文件格式分析")
print("="*60)

# 分析文件名格式
gt_feat_dir = 'output/sofa_data/language_features'
gt_files = sorted(glob.glob(f'{gt_feat_dir}/*.npy'))
print(f"\n文件数: {len(gt_files)}")
print("前20个文件:")
for f in gt_files[:20]:
    print(f"  {os.path.basename(f)}")

# _f 和 _s 可能代表 feature_level=1 和 feature_level=2
# segmentation目录02对应sofa_data图片02.jpg，索引为2
# 所以应该用 02_f.npy 或 02_s.npy

print("\n" + "="*60)
print("测试不同feature level")
print("="*60)

for suffix in ['_f.npy', '_s.npy']:
    gt_feat_path = f'output/sofa_data/language_features/02{suffix}'
    if os.path.exists(gt_feat_path):
        gt_feat = np.load(gt_feat_path)
        print(f"\n{gt_feat_path}:")
        print(f"  形状: {gt_feat.shape}")
        print(f"  范围: [{gt_feat.min():.4f}, {gt_feat.max():.4f}]")
        
        if gt_feat.shape[-1] == 512:
            gt_feat_norm = gt_feat / (np.linalg.norm(gt_feat, axis=-1, keepdims=True) + 1e-8)
            
            print(f"  GT feature + CLIP text 相似度:")
            for cat in ['Gundam', 'Pikachu', 'grey sofa']:
                text_tokens = tokenizer([cat]).to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(text_tokens)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat.cpu().numpy()[0]
                sim = np.dot(gt_feat_norm, text_feat)
                print(f"    '{cat}': max={sim.max():.4f}, >0.3: {(sim > 0.3).mean()*100:.1f}%")

# 检查language_features_dim3目录
print("\n" + "="*60)
print("检查 language_features_dim3 (编码后的3维特征)")
print("="*60)

dim3_dir = 'output/sofa_data/language_features_dim3'
dim3_files = sorted(glob.glob(f'{dim3_dir}/*.npy'))
print(f"文件数: {len(dim3_files)}")
print("前10个文件:", [os.path.basename(f) for f in dim3_files[:10]])

# 加载dim3特征
dim3_path = 'output/sofa_data/language_features_dim3/02_f.npy'
if os.path.exists(dim3_path):
    dim3_feat = np.load(dim3_path)
    print(f"\n{dim3_path}:")
    print(f"  形状: {dim3_feat.shape}")
    print(f"  范围: [{dim3_feat.min():.4f}, {dim3_feat.max():.4f}]")
    
    # 解码dim3特征
    dim3_flat = dim3_feat.reshape(-1, 3)
    dim3_tensor = torch.from_numpy(dim3_flat).float().to(device)
    with torch.no_grad():
        decoded_dim3 = ae.decode(dim3_tensor).cpu().numpy()
    decoded_dim3 = decoded_dim3.reshape(dim3_feat.shape[0], dim3_feat.shape[1], 512)
    decoded_dim3_norm = decoded_dim3 / (np.linalg.norm(decoded_dim3, axis=-1, keepdims=True) + 1e-8)
    
    print(f"  解码后与CLIP text相似度:")
    for cat in ['Gundam', 'Pikachu', 'grey sofa']:
        text_tokens = tokenizer([cat]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu().numpy()[0]
        sim = np.dot(decoded_dim3_norm, text_feat)
        print(f"    '{cat}': max={sim.max():.4f}, >0.3: {(sim > 0.3).mean()*100:.1f}%")

# 对比渲染特征和GT dim3特征
print("\n" + "="*60)
print("关键对比: 渲染特征 vs GT dim3特征")
print("="*60)

render_feat = np.load('output/sofa_1/train/ours_None/renders_npy/00002.npy')
print(f"渲染特征: {render_feat.shape}, 范围[{render_feat.min():.4f}, {render_feat.max():.4f}]")

if os.path.exists(dim3_path):
    print(f"GT dim3: {dim3_feat.shape}, 范围[{dim3_feat.min():.4f}, {dim3_feat.max():.4f}]")
    
    # 如果分辨率不同，缩放
    if dim3_feat.shape[:2] != render_feat.shape[:2]:
        scale_h = render_feat.shape[0] / dim3_feat.shape[0]
        scale_w = render_feat.shape[1] / dim3_feat.shape[1]
        dim3_scaled = zoom(dim3_feat, (scale_h, scale_w, 1), order=1)
        print(f"GT dim3缩放后: {dim3_scaled.shape}")
    else:
        dim3_scaled = dim3_feat
    
    # 像素级对比
    diff = np.abs(render_feat - dim3_scaled).mean()
    cos_sim = (render_feat * dim3_scaled).sum(axis=-1) / (np.linalg.norm(render_feat, axis=-1) * np.linalg.norm(dim3_scaled, axis=-1) + 1e-8)
    print(f"  L1差异: {diff:.4f}")
    print(f"  余弦相似度: mean={cos_sim.mean():.4f}, max={cos_sim.max():.4f}")
