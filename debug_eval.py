"""诊断评估问题"""
import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip
from PIL import Image, ImageDraw

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

# 加载AE
encoder_dims = [256, 128, 64, 32, 3]
decoder_dims = [16, 32, 64, 128, 256, 256, 512]
ae = Autoencoder(encoder_dims, decoder_dims)
ckpt = torch.load('autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth', map_location='cpu')
ae.load_state_dict(ckpt)
ae = ae.to(device)
ae.eval()

# 加载CLIP
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-16')

# 检查GT文件名
print("=== GT JSON文件名 ===")
json_files = sorted(glob.glob('output/sofa_data/segmentations_json_v3/frame_*.json'))
for jf in json_files:
    print(f"  {os.path.basename(jf)}")

print("\n=== 渲染文件名 ===")
npy_files = sorted(glob.glob('output/sofa_1/train/ours_None/renders_npy/*.npy'))
for nf in npy_files[:10]:
    print(f"  {os.path.basename(nf)}")

# 检查帧映射
print("\n=== 检查帧映射 ===")
# GT: frame_00003.json -> 想要第3张图(索引3)
# 但渲染是按顺序: 00000.npy=第0张, 00001.npy=第1张...
# 需要找出GT frame_00003对应哪张渲染图

# 检查GT图片和sofa_data图片的对应关系
print("GT图片名 vs sofa_data图片名:")
for jf in json_files:
    frame_id = int(os.path.basename(jf).replace('frame_', '').replace('.json', ''))
    # 尝试找到对应的图片
    img_name = f"{frame_id:05d}.jpg"  # 假设格式
    
    # 读取JSON获取实际图片名
    with open(jf) as f:
        data = json.load(f)
    actual_name = data.get('info', {}).get('name', 'unknown')
    print(f"  frame_id={frame_id}, JSON中图片名={actual_name}")

# 加载一个渲染特征检查
print("\n=== 检查渲染特征 ===")
feat = np.load('output/sofa_1/train/ours_None/renders_npy/00000.npy')
print(f"特征形状: {feat.shape}")
print(f"特征范围: [{feat.min():.4f}, {feat.max():.4f}]")
print(f"特征均值: {feat.mean():.4f}")

# 解码特征
feat_flat = feat.reshape(-1, 3)
feat_tensor = torch.from_numpy(feat_flat).float().to(device)
with torch.no_grad():
    decoded = ae.decode(feat_tensor)
decoded = decoded.cpu().numpy()
print(f"解码后形状: {decoded.shape}")
print(f"解码后范围: [{decoded.min():.4f}, {decoded.max():.4f}]")
print(f"解码后均值: {decoded.mean():.4f}")

# 检查解码特征的范数
norms = np.linalg.norm(decoded, axis=-1)
print(f"解码特征范数: mean={norms.mean():.4f}, std={norms.std():.4f}")

# 编码一个query
print("\n=== 检查CLIP编码 ===")
query = "grey sofa"
text_tokens = tokenizer([query]).to(device)
with torch.no_grad():
    text_feat = model.encode_text(text_tokens)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
text_feat = text_feat.cpu().numpy()[0]
print(f"Query: '{query}'")
print(f"CLIP特征形状: {text_feat.shape}")
print(f"CLIP特征范数: {np.linalg.norm(text_feat):.4f}")

# 计算相似度
decoded_reshaped = decoded.reshape(feat.shape[0], feat.shape[1], 512)
decoded_norm = decoded_reshaped / (np.linalg.norm(decoded_reshaped, axis=-1, keepdims=True) + 1e-8)
similarity = np.dot(decoded_norm, text_feat)
print(f"\n相似度分布:")
print(f"  min={similarity.min():.4f}, max={similarity.max():.4f}")
print(f"  mean={similarity.mean():.4f}, std={similarity.std():.4f}")
print(f"  >0.3的比例: {(similarity > 0.3).mean()*100:.2f}%")
print(f"  >0.2的比例: {(similarity > 0.2).mean()*100:.2f}%")
print(f"  >0.1的比例: {(similarity > 0.1).mean()*100:.2f}%")

# 检查GT mask
print("\n=== 检查GT mask ===")
with open('output/sofa_data/segmentations_json_v3/frame_00003.json') as f:
    data = json.load(f)
    
for i, obj in enumerate(data.get('objects', [])):
    cat = obj.get('category', '')
    seg = obj.get('segmentation', [])
    if seg:
        # 生成mask
        h = data['info']['height']
        w = data['info']['width']
        img = Image.new('L', (w, h), 0)
        flat_polygon = [(p[0], p[1]) for p in seg]
        ImageDraw.Draw(img).polygon(flat_polygon, fill=1)
        mask = np.array(img)
        print(f"对象{i}: '{cat}', mask非零像素={mask.sum()}, 占比={mask.mean()*100:.2f}%")
