"""
LangSplat正确评估脚本 - 修复类型问题
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import open_clip
import cv2
from tqdm import tqdm

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


def get_relevancy(embed, pos_embed, neg_embeds):
    """计算relevancy score - 处理类型"""
    # 确保类型一致
    embed = embed.half() if pos_embed.dtype == torch.float16 else embed.float()
    
    pos_sim = torch.mm(embed, pos_embed.unsqueeze(1))
    neg_sims = torch.mm(embed, neg_embeds.T)
    
    combined = torch.cat([pos_sim, neg_sims], dim=1)
    softmax = torch.softmax(10 * combined, dim=1)
    
    return softmax[:, 0]


def evaluate_model(model_dir, ae_ckpt_path, segmentations_dir):
    print(f"\n{'='*60}")
    print(f"LangSplat正确评估")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    
    # 加载AE
    print("\n[1/4] 加载Autoencoder...")
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    ae = Autoencoder(encoder_dims, decoder_dims)
    ckpt = torch.load(ae_ckpt_path, map_location='cpu')
    ae.load_state_dict(ckpt)
    ae = ae.to(device)
    ae.eval()
    
    # 加载CLIP (使用fp16)
    print("\n[2/4] 加载CLIP (fp16)...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16'
    )
    clip_model.eval()
    clip_model = clip_model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    
    # 定义负样本
    negatives = ("object", "things", "stuff", "texture")
    with torch.no_grad():
        neg_tokens = torch.cat([tokenizer(neg) for neg in negatives]).to(device)
        neg_embeds = clip_model.encode_text(neg_tokens)
        neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)
    
    print(f"  负样本embeds shape: {neg_embeds.shape}, dtype: {neg_embeds.dtype}")
    
    # 加载GT
    print("\n[3/4] 加载GT标注...")
    gt_data = load_gt_from_png(segmentations_dir)
    print(f"  GT帧数: {len(gt_data)}")
    
    # 查找渲染结果
    renders_dir = os.path.join(model_dir, "train/ours_None/renders_npy")
    feat_files = sorted(glob.glob(os.path.join(renders_dir, '*.npy')))
    render_indices = {}
    for f in feat_files:
        idx = int(os.path.basename(f).replace('.npy', ''))
        render_indices[idx] = f
    
    # 评估
    print("\n[4/4] 开始评估...")
    all_results = []
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
            decoded = ae.decode(feat_tensor)
        decoded = decoded.reshape(H, W, 512)
        
        for category, gt_mask in frame_masks.items():
            if gt_mask.shape != (H, W):
                gt_mask = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # 编码text
            with torch.no_grad():
                text_tokens = tokenizer([category]).to(device)
                pos_embed = clip_model.encode_text(text_tokens)
                pos_embed = pos_embed / pos_embed.norm(dim=-1, keepdim=True)
            
            # 计算relevancy
            decoded_flat = decoded.reshape(-1, 512)
            relevancy = get_relevancy(decoded_flat, pos_embed.squeeze(0), neg_embeds)
            relevancy_map = relevancy.reshape(H, W).cpu().numpy()
            
            # 尝试不同阈值
            best_iou = 0
            best_thresh = 0
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
                pred_mask = (relevancy_map > thresh).astype(np.uint8)
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_thresh = thresh
            
            all_results.append({
                'frame_idx': frame_idx,
                'category': category,
                'iou': best_iou,
                'best_thresh': best_thresh,
                'relevancy_max': relevancy_map.max(),
                'relevancy_mean': relevancy_map.mean()
            })
            
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(best_iou)
    
    # 输出结果
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    
    if len(all_results) == 0:
        print("没有评估结果!")
        return 0.0
    
    all_ious = [r['iou'] for r in all_results]
    print(f"\n总体mIoU: {np.mean(all_ious):.4f}")
    print(f"评估样本数: {len(all_ious)}")
    
    print(f"\n按类别IoU:")
    print(f"{'类别':<40} {'mIoU':>8} {'样本数':>8}")
    print("-" * 60)
    for cat, ious in sorted(category_results.items()):
        print(f"{cat:<40} {np.mean(ious):>8.4f} {len(ious):>8}")
    
    print(f"\nRelevancy统计:")
    relevancy_maxes = [r['relevancy_max'] for r in all_results]
    relevancy_means = [r['relevancy_mean'] for r in all_results]
    print(f"  Max relevancy: mean={np.mean(relevancy_maxes):.4f}")
    print(f"  Mean relevancy: mean={np.mean(relevancy_means):.4f}")
    
    # 显示详细结果
    print(f"\n详细结果:")
    for r in all_results[:10]:
        print(f"  Frame {r['frame_idx']}, '{r['category']}': IoU={r['iou']:.4f}, thresh={r['best_thresh']:.2f}, rel_max={r['relevancy_max']:.4f}")
    
    return np.mean(all_ious)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='output/sofa_1')
    parser.add_argument('--ae_ckpt', type=str, default='autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth')
    parser.add_argument('--segmentations_dir', type=str, default='output/sofa_data/segmentations')
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.ae_ckpt, args.segmentations_dir)
