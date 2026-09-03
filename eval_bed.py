#!/usr/bin/env python3
"""
Bed场景评估脚本 - 使用relevancy score方法
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import json

# 设置路径
model_path = 'output/bed_output_-1'
data_path = 'output/bed_data'
iteration = 30000

device = 'cuda'

# 加载checkpoint
print(f"加载checkpoint: {model_path}/chkpnt{iteration}.pth")
checkpoint = torch.load(f'{model_path}/chkpnt{iteration}.pth', map_location=device)

print(f"Checkpoint长度: {len(checkpoint)}")

# 解析checkpoint
active_sh_degree = checkpoint[0]
_xyz = checkpoint[1]
_language_feature = checkpoint[7] if len(checkpoint) > 7 else None

print(f"点数: {_xyz.shape[0]}")
if _language_feature is not None:
    print(f"语言特征维度: {_language_feature.shape}")
else:
    print("警告: 无语言特征!")
    sys.exit(1)

# 加载CLIP模型
print("\n加载CLIP模型...")
import clip
clip_model, preprocess = clip.load('ViT-B/16', device=device)

# 定义bed场景的类别
categories = {
    'red bag': 'a red bag',
    'black leather shoe': 'a black leather shoe', 
    'banana': 'a banana',
    'hand': 'a hand',
    'camera': 'a camera',
    'white sheet': 'a white sheet'
}

# 编码查询文本
print("\n编码查询文本...")
text_embeddings = {}
for cat_name, query in categories.items():
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    text_embeddings[cat_name] = text_feat
    print(f"  {cat_name}: {query}")

# 处理语言特征
print("\n处理语言特征...")
lang_feat = _language_feature

if lang_feat.dim() == 2 and lang_feat.shape[1] == 3:
    print("检测到dim3特征，解码中...")
    ae_ckpt_path = 'autoencoder/ckpt/bed/ae_ckpt/best_ckpt.pth'
    if os.path.exists(ae_ckpt_path):
        sys.path.insert(0, 'autoencoder')
        from network import AutoDecoder
        ae = AutoDecoder(latent_dim=3, output_dim=512).to(device)
        ae.load_state_dict(torch.load(ae_ckpt_path, map_location=device))
        ae.eval()
        
        with torch.no_grad():
            lang_feat_512 = ae(lang_feat)
            lang_feat_512 = lang_feat_512 / lang_feat_512.norm(dim=-1, keepdim=True)
        print(f"解码后特征维度: {lang_feat_512.shape}")
        lang_feat = lang_feat_512
    else:
        print(f"错误: AE checkpoint不存在")
        sys.exit(1)

# 计算relevancy scores
print("\n计算relevancy scores...")
relevancy_scores = {}
for cat_name, text_feat in text_embeddings.items():
    similarity = (lang_feat @ text_feat.T).squeeze()
    relevancy_scores[cat_name] = similarity.cpu().numpy()
    print(f"  {cat_name}: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")

# 加载相机参数
with open(f'{model_path}/cameras.json', 'r') as f:
    cameras = json.load(f)

print(f"\n相机数量: {len(cameras)}")

# 创建图片名到相机参数的映射
cam_by_name = {}
for cam in cameras:
    img_name = cam['img_name']
    # 去掉扩展名
    base_name = os.path.splitext(img_name)[0]
    cam_by_name[base_name] = cam

# 加载分割标注
seg_dir = f'{data_path}/segmentations'
seg_frames = sorted([d for d in os.listdir(seg_dir) if os.path.isdir(os.path.join(seg_dir, d))])
print(f"分割帧: {seg_frames}")

# 评估IoU
print("\n" + "="*60)
print("评估IoU")
print("="*60)

# 加载渲染的语言特征图
render_npy_dir = f'{model_path}/test/ours_None/renders_npy'
if not os.path.exists(render_npy_dir):
    print(f"警告: 渲染目录不存在: {render_npy_dir}")
    render_npy_dir = f'{model_path}/train/ours_None/renders_npy'

render_files = sorted([f for f in os.listdir(render_npy_dir) if f.endswith('.npy')])
print(f"渲染文件数: {len(render_files)}")

# 计算每个类别的IoU
all_ious = {cat: [] for cat in categories.keys()}

for frame in seg_frames:
    frame_int = int(frame)
    frame_str = f'{frame_int:05d}'
    
    # 找到对应的渲染文件
    render_file = f'{frame_str}.npy'
    render_path = os.path.join(render_npy_dir, render_file)
    
    if not os.path.exists(render_path):
        print(f"  跳过帧 {frame}: 无渲染文件 {render_file}")
        continue
    
    # 加载渲染的语言特征图
    rendered_feat = np.load(render_path)  # [H, W, 3] or [H, W, C]
    H, W = rendered_feat.shape[:2]
    
    # 转换为tensor
    rendered_feat_t = torch.from_numpy(rendered_feat).float().to(device)
    
    # 如果是dim3特征，解码
    if rendered_feat_t.shape[-1] == 3:
        with torch.no_grad():
            rendered_feat_flat = rendered_feat_t.reshape(-1, 3)
            rendered_feat_512 = ae(rendered_feat_flat)
            rendered_feat_512 = rendered_feat_512 / rendered_feat_512.norm(dim=-1, keepdim=True)
            rendered_feat_512 = rendered_feat_512.reshape(H, W, 512)
    else:
        rendered_feat_t = rendered_feat_t / (rendered_feat_t.norm(dim=-1, keepdim=True) + 1e-8)
        rendered_feat_512 = rendered_feat_t
    
    # 评估每个类别
    frame_path = os.path.join(seg_dir, frame)
    for cat_name in categories.keys():
        mask_file = f'{cat_name}.png'
        mask_path = os.path.join(frame_path, mask_file)
        
        if not os.path.exists(mask_path):
            continue
        
        # 加载GT mask
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_binary = (gt_mask > 127).astype(np.uint8)
        
        if gt_binary.sum() == 0 or gt_binary.sum() == gt_binary.size:
            continue
        
        # 计算预测mask (使用relevancy score)
        text_feat = text_embeddings[cat_name]
        
        # 计算每个像素的相似度
        with torch.no_grad():
            similarity_map = (rendered_feat_512 @ text_feat.T).squeeze().cpu().numpy()
        
        # 使用阈值分割
        threshold = similarity_map.mean() + 0.5 * similarity_map.std()
        pred_mask = (similarity_map > threshold).astype(np.uint8)
        
        # 计算IoU
        intersection = (pred_mask & gt_binary).sum()
        union = (pred_mask | gt_binary).sum()
        
        if union > 0:
            iou = intersection / union
            all_ious[cat_name].append(iou)
            print(f"  帧{frame} {cat_name}: IoU={iou:.4f}")

# 计算mIoU
print("\n" + "="*60)
print("评估结果")
print("="*60)

miou_list = []
for cat_name, ious in all_ious.items():
    if ious:
        mean_iou = np.mean(ious)
        miou_list.append(mean_iou)
        print(f"{cat_name}: mIoU={mean_iou:.4f} ({len(ious)}帧)")
    else:
        print(f"{cat_name}: 无评估数据")

if miou_list:
    overall_miou = np.mean(miou_list)
    print(f"\n总体mIoU: {overall_miou:.4f}")

# 保存结果
results = {
    'per_category_miou': {k: float(np.mean(v)) if v else 0.0 for k, v in all_ious.items()},
    'overall_miou': float(np.mean(miou_list)) if miou_list else 0.0,
    'num_points': int(_xyz.shape[0]),
}

output_file = f'{model_path}/eval_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n结果已保存: {output_file}")
