"""诊断GT feature和渲染质量"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip

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
print("关键诊断：GT feature vs 渲染特征")
print("="*60)

# 1. 检查GT feature文件
print("\n[1] GT language features详情:")
gt_feat_dir = 'output/sofa_data/language_features'
gt_files = sorted(glob.glob(f'{gt_feat_dir}/*.npy'))[:10]
print(f"前10个文件: {[os.path.basename(f) for f in gt_files]}")

# 加载GT feature (应该对应segmentation目录02)
gt_feat_path = 'output/sofa_data/language_features/02.npy'
gt_feat = np.load(gt_feat_path)
print(f"\nGT feature 02.npy:")
print(f"  形状: {gt_feat.shape}")
print(f"  范围: [{gt_feat.min():.4f}, {gt_feat.max():.4f}]")

# 2. 用GT feature直接计算相似度
print("\n[2] GT feature + CLIP text 相似度:")
gt_feat_norm = gt_feat / (np.linalg.norm(gt_feat, axis=-1, keepdims=True) + 1e-8)

categories = ['Gundam', 'Pikachu', 'Xbox wireless controller', 
              'a stack of UNO cards', 'a red Nintendo Switch joy-con controller', 'grey sofa']
for cat in categories:
    text_tokens = tokenizer([cat]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().numpy()[0]
    
    similarity = np.dot(gt_feat_norm, text_feat)
    print(f"  '{cat}': max={similarity.max():.4f}, mean={similarity.mean():.4f}, >0.3: {(similarity > 0.3).mean()*100:.1f}%")

# 3. 检查是否需要不同的CLIP模型
print("\n[3] 尝试其他CLIP模型:")
# LangSplat可能用的是不同的CLIP
print("检查train.py中使用的CLIP模型...")

# 4. 对比GT feature编码-解码后的结果
print("\n[4] GT feature -> AE encode -> decode 测试:")
# 先编码GT feature为3维
gt_flat = gt_feat_norm.reshape(-1, 512)
gt_tensor = torch.from_numpy(gt_flat).float().to(device)

with torch.no_grad():
    # 编码
    encoded = ae.encode(gt_tensor)
    # 解码
    decoded = ae.decode(encoded)

decoded = decoded.cpu().numpy()
print(f"编码后形状: {encoded.shape}")  # 应该是 Nx3
print(f"解码后形状: {decoded.shape}")  # 应该是 Nx512

# 计算重建误差
recon_sim = (gt_flat * decoded).sum(axis=-1).mean()
print(f"重建相似度: {recon_sim:.4f}")

# 5. 渲染特征是否正确解码
print("\n[5] 渲染特征解码验证:")
render_feat = np.load('output/sofa_1/train/ours_None/renders_npy/00002.npy')
print(f"渲染特征原始: {render_feat.shape}, 范围[{render_feat.min():.4f}, {render_feat.max():.4f}]")

# 解码
render_flat = render_feat.reshape(-1, 3)
render_tensor = torch.from_numpy(render_flat).float().to(device)
with torch.no_grad():
    decoded_render = ae.decode(render_tensor)
decoded_render = decoded_render.cpu().numpy().reshape(render_feat.shape[0], render_feat.shape[1], 512)
decoded_render_norm = decoded_render / (np.linalg.norm(decoded_render, axis=-1, keepdims=True) + 1e-8)

# 与GT feature对比 (需要缩放)
from scipy.ndimage import zoom
if gt_feat_norm.shape[:2] != decoded_render_norm.shape[:2]:
    scale_h = decoded_render_norm.shape[0] / gt_feat_norm.shape[0]
    scale_w = decoded_render_norm.shape[1] / gt_feat_norm.shape[1]
    gt_scaled = zoom(gt_feat_norm, (scale_h, scale_w, 1), order=1)
else:
    gt_scaled = gt_feat_norm

pixel_sim = (gt_scaled * decoded_render_norm).sum(axis=-1)
print(f"GT vs 渲染特征(解码后) 像素级相似度:")
print(f"  mean={pixel_sim.mean():.4f}, max={pixel_sim.max():.4f}, min={pixel_sim.min():.4f}")

# 6. 检查训练时使用的CLIP
print("\n[6] 检查训练配置:")
import glob
cfg_files = glob.glob('output/sofa_1/cfg_args')
if cfg_files:
    with open(cfg_files[0]) as f:
        print(f.read())
