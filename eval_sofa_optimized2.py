"""
LangSplat优化评估脚本 - 修复版
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip
import cv2
from tqdm import tqdm
from skimage import filters
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
    gt_data = {}
    for subdir in sorted(os.listdir(segmentations_dir)):
        subdir_path = os.path.join(segmentations_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        try:
            dir_num = int(subdir)
            frame_idx = dir_num
        except ValueError:
            continue
        
        frame_masks = {}
        for mask_file in glob.glob(os.path.join(subdir_path, '*.png')):
            category = os.path.basename(mask_file).replace('.png', '')
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask_scaled = cv2.resize(mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_scaled > 127).astype(np.uint8)
            mask_ratio = mask_binary.mean()
            if mask_ratio > 0.5:
                continue
            frame_masks[category] = mask_binary
        
        if frame_masks:
            gt_data[frame_idx] = frame_masks
    return gt_data


def get_relevancy_map(decoded, pos_embed, neg_embeds, temp=20):
    """计算relevancy map"""
    H, W, _ = decoded.shape
    decoded_flat = decoded.reshape(-1, 512)
    
    # 确保pos_embed是1D向量
    if pos_embed.dim() > 1:
        pos_embed = pos_embed.reshape(-1)
    
    pos_sim = torch.mv(decoded_flat, pos_embed)  # 使用mv代替mm
    neg_sims = torch.mm(decoded_flat, neg_embeds.T)
    combined = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
    softmax = torch.softmax(temp * combined, dim=1)
    relevancy = softmax[:, 0].reshape(H, W).cpu().numpy()
    
    return relevancy


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0


def post_process_mask(mask, min_area=100):
    """后处理：去除小区域"""
    labeled, num_features = ndimage.label(mask)
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < min_area:
            mask[labeled == i] = 0
    return mask


def evaluate_model(model_dir, ae_ckpt_path, segmentations_dir):
    print(f"\n{'='*60}")
    print(f"LangSplat优化评估")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    
    # 加载AE
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    ae = Autoencoder(encoder_dims, decoder_dims)
    ckpt = torch.load(ae_ckpt_path, map_location='cpu')
    ae.load_state_dict(ckpt)
    ae = ae.to(device)
    ae.eval()
    
    # 加载CLIP
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16'
    )
    clip_model.eval()
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    
    # 负样本
    negatives = ("object", "things", "stuff", "texture")
    with torch.no_grad():
        neg_tokens = torch.cat([tokenizer(neg) for neg in negatives]).to(device)
        neg_embeds = clip_model.encode_text(neg_tokens)
        neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
    
    # 加载GT
    gt_data = load_gt_from_png(segmentations_dir)
    
    # 查找渲染结果
    renders_dir = os.path.join(model_dir, "train/ours_None/renders_npy")
    feat_files = sorted(glob.glob(os.path.join(renders_dir, '*.npy')))
    render_indices = {}
    for f in feat_files:
        idx = int(os.path.basename(f).replace('.npy', ''))
        render_indices[idx] = f
    
    # 评估
    results_by_method = {
        'thresh_0.15': [],
        'thresh_0.10': [],
        'otsu': [],
        'adaptive': [],
        'best_overall': []
    }
    category_results = {}
    
    for frame_idx, frame_masks in tqdm(gt_data.items(), desc="评估帧"):
        if frame_idx not in render_indices:
            continue
        
        feat_path = render_indices[frame_idx]
        compressed_feat = np.load(feat_path)
        H, W, _ = compressed_feat.shape
        
        # 解码
        feat_flat = compressed_feat.reshape(-1, 3)
        feat_tensor = torch.from_numpy(feat_flat).float().to(device)
        with torch.no_grad():
            decoded = ae.decode(feat_tensor).half()
        decoded = decoded.reshape(H, W, 512)
        
        for category, gt_mask in frame_masks.items():
            if gt_mask.shape != (H, W):
                gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # 编码text
            with torch.no_grad():
                text_tokens = tokenizer([category]).to(device)
                pos_embed = clip_model.encode_text(text_tokens)
                pos_embed = pos_embed / pos_embed.norm(dim=-1, keepdim=True)
                pos_embed = pos_embed.reshape(-1)  # 确保是1D
            
            # 计算relevancy (temp=20分离度更好)
            relevancy = get_relevancy_map(decoded, pos_embed, neg_embeds, temp=20)
            
            # 方法1: 固定阈值0.15
            pred_015 = post_process_mask((relevancy > 0.15).astype(np.uint8))
            iou_015 = compute_iou(pred_015, gt_mask)
            results_by_method['thresh_0.15'].append(iou_015)
            
            # 方法2: 固定阈值0.10
            pred_010 = post_process_mask((relevancy > 0.10).astype(np.uint8))
            iou_010 = compute_iou(pred_010, gt_mask)
            results_by_method['thresh_0.10'].append(iou_010)
            
            # 方法3: Otsu阈值
            try:
                otsu_thresh = filters.threshold_otsu(relevancy)
                pred_otsu = post_process_mask((relevancy > otsu_thresh).astype(np.uint8))
                iou_otsu = compute_iou(pred_otsu, gt_mask)
            except:
                iou_otsu = 0
            results_by_method['otsu'].append(iou_otsu)
            
            # 方法4: 自适应阈值 (percentile)
            try:
                perc_thresh = np.percentile(relevancy, 90)
                pred_adapt = post_process_mask((relevancy > perc_thresh).astype(np.uint8))
                iou_adapt = compute_iou(pred_adapt, gt_mask)
            except:
                iou_adapt = 0
            results_by_method['adaptive'].append(iou_adapt)
            
            # 方法5: 最佳阈值扫描
            best_iou = max(iou_015, iou_010, iou_otsu, iou_adapt)
            for thresh in np.arange(0.08, 0.25, 0.02):
                pred = (relevancy > thresh).astype(np.uint8)
                iou = compute_iou(pred, gt_mask)
                if iou > best_iou:
                    best_iou = iou
            results_by_method['best_overall'].append(best_iou)
            
            # 记录类别结果
            if category not in category_results:
                category_results[category] = {'thresh_0.15': [], 'best': []}
            category_results[category]['thresh_0.15'].append(iou_015)
            category_results[category]['best'].append(best_iou)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    
    print(f"\n{'方法':<20} {'mIoU':>10} {'样本数':>10}")
    print("-" * 45)
    for method, ious in results_by_method.items():
        print(f"{method:<20} {np.mean(ious):>10.4f} {len(ious):>10}")
    
    print(f"\n按类别mIoU (阈值0.15):")
    print(f"{'类别':<40} {'mIoU':>10} {'样本数':>10}")
    print("-" * 65)
    for cat, results in sorted(category_results.items()):
        print(f"{cat:<40} {np.mean(results['thresh_0.15']):>10.4f} {len(results['thresh_0.15']):>10}")
    
    print(f"\n按类别mIoU (最佳阈值):")
    print(f"{'类别':<40} {'mIoU':>10} {'样本数':>10}")
    print("-" * 65)
    for cat, results in sorted(category_results.items()):
        print(f"{cat:<40} {np.mean(results['best']):>10.4f} {len(results['best']):>10}")
    
    return np.mean(results_by_method['thresh_0.15'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='output/sofa_1')
    parser.add_argument('--ae_ckpt', type=str, default='autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth')
    parser.add_argument('--segmentations_dir', type=str, default='output/sofa_data/segmentations')
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.ae_ckpt, args.segmentations_dir)
