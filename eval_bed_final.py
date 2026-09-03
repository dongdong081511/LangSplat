#!/usr/bin/env python3
"""
Bed场景评估脚本
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import json

model_path = 'output/bed_output_lang_0'
data_path = 'output/bed_data'
iteration = 30000
device = 'cuda'

print(f"加载模型: {model_path}")

# 加载checkpoint
checkpoint = torch.load(f'{model_path}/chkpnt{iteration}.pth', map_location=device)
model_params, first_iter = checkpoint

_xyz = model_params[1]
_language_feature = model_params[7]

print(f"点数: {_xyz.shape[0]}")
print(f"语言特征维度: {_language_feature.shape}")

# 加载CLIP
print("\n加载CLIP模型...")
import clip
clip_model, preprocess = clip.load('ViT-B/16', device=device)

# bed场景类别
categories = {
    'red bag': 'a red bag',
    'black leather shoe': 'a black leather shoe', 
    'banana': 'a banana',
    'hand': 'a hand',
    'camera': 'a camera',
    'white sheet': 'a white sheet'
}

print("编码查询文本...")
text_embeddings = {}
for cat_name, query in categories.items():
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text)
        text_feat = text_feat.float()  # 转换为float32
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_embeddings[cat_name] = text_feat

# 处理语言特征
lang_feat = _language_feature.float()

# 加载AE
ae_ckpt_path = 'autoencoder/ckpt/bed/ae_ckpt/best_ckpt.pth'
if os.path.exists(ae_ckpt_path) and lang_feat.shape[1] == 3:
    print("解码dim3特征...")
    
    sys.path.insert(0, 'autoencoder')
    from model import Autoencoder
    
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    
    ae = Autoencoder(encoder_dims, decoder_dims).to(device)
    ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
    ae.eval()
    
    with torch.no_grad():
        lang_feat_512 = ae.decode(lang_feat)
        lang_feat_512 = lang_feat_512 / lang_feat_512.norm(dim=-1, keepdim=True)
    lang_feat = lang_feat_512.float()
    print(f"解码后维度: {lang_feat.shape}")

# 计算relevancy scores
print("\n计算relevancy scores...")
for cat_name, text_feat in text_embeddings.items():
    similarity = (lang_feat @ text_feat.T).squeeze()
    print(f"  {cat_name}: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")

# 加载相机
with open(f'{model_path}/cameras.json', 'r') as f:
    cameras = json.load(f)

train_imgs = sorted([cam['img_name'] for cam in cameras])
print(f"\n训练图片数: {len(train_imgs)}")

# 加载分割
seg_dir = f'{data_path}/segmentations'
seg_frames = sorted([d for d in os.listdir(seg_dir) if os.path.isdir(os.path.join(seg_dir, d))])
print(f"分割帧: {seg_frames}")

# 渲染文件
render_dir = f'{model_path}/train/ours_None/renders_npy'
render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.npy')])
print(f"渲染文件数: {len(render_files)}")

# 评估
print("\n" + "="*60)
print("评估IoU")
print("="*60)

all_ious = {cat: [] for cat in categories.keys()}

for frame in seg_frames:
    matching_idx = None
    for idx, img_name in enumerate(train_imgs):
        img_base = os.path.splitext(img_name)[0]
        if img_base == frame.zfill(2) or img_base == frame:
            matching_idx = idx
            break
    
    if matching_idx is None:
        try:
            matching_idx = int(frame)
        except:
            continue
    
    render_file = f'{matching_idx:05d}.npy'
    render_path = os.path.join(render_dir, render_file)
    
    if not os.path.exists(render_path):
        print(f"帧{frame}: 无渲染文件 {render_file}")
        continue
    
    print(f"\n处理帧 {frame} (索引{matching_idx})")
    
    rendered_feat = np.load(render_path)
    H, W = rendered_feat.shape[:2]
    rendered_feat_t = torch.from_numpy(rendered_feat).float().to(device)
    
    if rendered_feat_t.shape[-1] == 3:
        with torch.no_grad():
            feat_flat = rendered_feat_t.reshape(-1, 3)
            feat_512 = ae.decode(feat_flat)
            feat_512 = feat_512 / feat_512.norm(dim=-1, keepdim=True)
            feat_512 = feat_512.reshape(H, W, 512).float()
    else:
        feat_512 = rendered_feat_t
    
    frame_path = os.path.join(seg_dir, frame)
    for cat_name in categories.keys():
        mask_file = f'{cat_name}.png'
        mask_path = os.path.join(frame_path, mask_file)
        
        if not os.path.exists(mask_path):
            continue
        
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        if gt_binary.shape[:2] != (H, W):
            gt_binary = np.array(Image.fromarray(gt_binary).resize((W, H), Image.NEAREST))
        
        if gt_binary.sum() == 0 or gt_binary.sum() == gt_binary.size:
            continue
        
        text_feat = text_embeddings[cat_name]
        with torch.no_grad():
            sim_map = (feat_512 @ text_feat.T).squeeze().cpu().numpy()
        
        thresh = sim_map.mean() + 0.5 * sim_map.std()
        pred_mask = (sim_map > thresh).astype(np.uint8)
        
        inter = (pred_mask & gt_binary).sum()
        union = (pred_mask | gt_binary).sum()
        iou = inter / union if union > 0 else 0
        
        all_ious[cat_name].append(iou)
        print(f"  {cat_name}: IoU={iou:.4f}")

# 结果
print("\n" + "="*60)
print("结果")
print("="*60)

miou_list = []
for cat, ious in all_ious.items():
    if ious:
        miou = np.mean(ious)
        miou_list.append(miou)
        print(f"{cat}: mIoU={miou:.4f} ({len(ious)}帧)")
    else:
        print(f"{cat}: 无数据")

if miou_list:
    print(f"\n总体mIoU: {np.mean(miou_list):.4f}")

results = {
    'per_category_miou': {k: float(np.mean(v)) if v else 0.0 for k, v in all_ious.items()},
    'overall_miou': float(np.mean(miou_list)) if miou_list else 0.0,
}
with open(f'{model_path}/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n已保存: {model_path}/eval_results.json")
