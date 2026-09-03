"""
LangSplat 3D-OVS 评估脚本 (修正版)
正确处理帧映射和排除背景类
"""
import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from tqdm import tqdm
import open_clip
import cv2

# ========== Autoencoder模型定义 ==========
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


def load_gt_from_png(segmentations_dir, render_h=1080, render_w=1440):
    """
    直接从原始PNG mask加载GT标注
    返回: {frame_idx: {category: mask}}
    """
    gt_data = {}
    
    # 遍历每个子目录 (02, 04, 10, 15, 22)
    for subdir in sorted(os.listdir(segmentations_dir)):
        subdir_path = os.path.join(segmentations_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # 目录名转换为frame_idx
        # 02 -> frame_00003 (2+1=3)
        try:
            dir_num = int(subdir)
            frame_idx = dir_num  # 使用目录号作为sofa_data中的图片索引
        except ValueError:
            continue
        
        frame_masks = {}
        
        # 读取每个类别的mask
        for mask_file in glob.glob(os.path.join(subdir_path, '*.png')):
            category = os.path.basename(mask_file).replace('.png', '')
            
            # 读取mask
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            # 缩放到渲染分辨率
            mask_scaled = cv2.resize(mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_scaled > 127).astype(np.uint8)
            
            # 计算mask占比，排除占比过大的背景类
            mask_ratio = mask_binary.mean()
            if mask_ratio > 0.5:  # 超过50%像素认为是背景类
                print(f"  跳过背景类 '{category}' (占比{mask_ratio*100:.1f}%)")
                continue
            
            frame_masks[category] = mask_binary
        
        if frame_masks:
            gt_data[frame_idx] = frame_masks
    
    return gt_data


def compute_iou(pred_mask, gt_mask):
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_model(model_dir, ae_ckpt_path, segmentations_dir, 
                   clip_model_name='ViT-B-16', 
                   clip_pretrained='laion2b_s34b_b88k',
                   similarity_threshold=0.2):
    """评估模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_dir}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # ========== 加载Autoencoder ==========
    print("\n[1/4] 加载Autoencoder...")
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    ae = Autoencoder(encoder_dims, decoder_dims)
    ckpt = torch.load(ae_ckpt_path, map_location='cpu')
    ae.load_state_dict(ckpt)
    ae = ae.to(device)
    ae.eval()
    print(f"  Autoencoder加载完成")
    
    # ========== 加载CLIP ==========
    print("\n[2/4] 加载CLIP模型...")
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    print(f"  CLIP模型: {clip_model_name}")
    
    # ========== 加载GT标注 ==========
    print("\n[3/4] 加载GT标注 (从PNG)...")
    gt_data = load_gt_from_png(segmentations_dir)
    print(f"  GT帧数: {len(gt_data)}")
    for frame_idx, masks in gt_data.items():
        print(f"    Frame {frame_idx}: {len(masks)} 个对象 - {list(masks.keys())}")
    
    # ========== 查找渲染结果 ==========
    renders_dir = os.path.join(model_dir, "train/ours_None/renders_npy")
    feat_files = sorted(glob.glob(os.path.join(renders_dir, '*.npy')))
    print(f"\n  渲染特征文件数: {len(feat_files)}")
    
    # 建立索引映射
    render_indices = {}
    for f in feat_files:
        idx = int(os.path.basename(f).replace('.npy', ''))
        render_indices[idx] = f
    
    # ========== 评估 ==========
    print(f"\n[4/4] 开始评估 (阈值={similarity_threshold})...")
    
    all_results = []
    category_results = {}
    
    for frame_idx, frame_masks in tqdm(gt_data.items(), desc="评估帧"):
        # 查找对应的渲染文件
        if frame_idx not in render_indices:
            print(f"  警告: Frame {frame_idx} 没有对应的渲染结果")
            continue
        
        feat_path = render_indices[frame_idx]
        
        # 加载渲染的语言特征
        compressed_feat = np.load(feat_path)  # HxWx3
        H, W, _ = compressed_feat.shape
        
        # 解码特征
        feat_flat = compressed_feat.reshape(-1, 3)
        feat_tensor = torch.from_numpy(feat_flat).float().to(device)
        
        with torch.no_grad():
            decoded_feat = ae.decode(feat_tensor)
        decoded_feat = decoded_feat.cpu().numpy()
        decoded_feat = decoded_feat.reshape(H, W, 512)
        
        # 归一化
        feat_norm = decoded_feat / (np.linalg.norm(decoded_feat, axis=-1, keepdims=True) + 1e-8)
        
        # 评估每个类别
        for category, gt_mask in frame_masks.items():
            # 确保尺寸匹配
            if gt_mask.shape != (H, W):
                gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # 编码query
            text_tokens = tokenizer([category]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feat = text_features.cpu().numpy()[0]
            
            # 计算相似度
            similarity = np.dot(feat_norm, text_feat)
            
            # 生成预测mask
            pred_mask = (similarity > similarity_threshold).astype(np.uint8)
            
            # 计算IoU
            iou = compute_iou(pred_mask, gt_mask)
            
            all_results.append({
                'frame_idx': frame_idx,
                'category': category,
                'iou': iou,
                'pred_pixels': pred_mask.sum(),
                'gt_pixels': gt_mask.sum()
            })
            
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(iou)
    
    # ========== 输出结果 ==========
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    
    if len(all_results) == 0:
        print("警告: 没有成功评估任何样本!")
        return 0.0
    
    # 总体mIoU
    all_ious = [r['iou'] for r in all_results]
    miou = np.mean(all_ious)
    print(f"\n总体mIoU: {miou:.4f}")
    print(f"评估样本数: {len(all_ious)}")
    
    # 按类别输出
    print(f"\n按类别IoU:")
    print(f"{'类别':<40} {'mIoU':>8} {'样本数':>8}")
    print("-" * 60)
    for cat, ious in sorted(category_results.items()):
        cat_miou = np.mean(ious)
        print(f"{cat:<40} {cat_miou:>8.4f} {len(ious):>8}")
    
    # 输出详细结果
    print(f"\n详细结果 (前10条):")
    for r in all_results[:10]:
        print(f"  Frame {r['frame_idx']}, '{r['category']}': IoU={r['iou']:.4f}, pred={r['pred_pixels']}, gt={r['gt_pixels']}")
    
    return miou


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LangSplat 3D-OVS评估 (修正版)')
    parser.add_argument('--model_dir', type=str, default='output/sofa_1')
    parser.add_argument('--ae_ckpt', type=str, default='autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth')
    parser.add_argument('--segmentations_dir', type=str, default='output/sofa_data/segmentations')
    parser.add_argument('--threshold', type=float, default=0.2, help='相似度阈值')
    args = parser.parse_args()
    
    evaluate_model(
        args.model_dir, 
        args.ae_ckpt, 
        args.segmentations_dir,
        similarity_threshold=args.threshold
    )
