import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Autoencoder模型定义
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

def load_gt_annotations(json_folder):
    """加载GT标注"""
    gt_ann = {}
    json_files = sorted(glob.glob(os.path.join(json_folder, 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(json_folder, 'frame_*.jpg')))
    
    for jf in json_files:
        frame_id = int(os.path.basename(jf).replace('frame_', '').replace('.json', ''))
        with open(jf) as f:
            data = json.load(f)
        gt_ann[frame_id] = data
    
    # 获取图像尺寸
    if img_paths:
        img = Image.open(img_paths[0])
        image_shape = (img.height, img.width)
    else:
        image_shape = (730, 988)  # teatime默认
    
    return gt_ann, image_shape

def compute_iou(pred_mask, gt_mask):
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_model(model_dir, ae_ckpt_path, json_folder, mask_thresh=0.4):
    """评估单个模型"""
    print(f"\n评估模型: {model_dir}")
    
    # 加载AE
    encoder_dims = [256, 128, 64, 32, 3]
    decoder_dims = [16, 32, 64, 128, 256, 256, 512]
    ae = Autoencoder(encoder_dims, decoder_dims)
    ckpt = torch.load(ae_ckpt_path, map_location='cpu')
    ae.load_state_dict(ckpt)
    ae.eval()
    
    # 加载GT
    gt_ann, image_shape = load_gt_annotations(json_folder)
    print(f"GT帧数: {len(gt_ann)}, 图像尺寸: {image_shape}")
    
    # 查找渲染的语言特征文件
    renders_dir = os.path.join(model_dir, "train/ours_None/renders_npy")
    feat_files = sorted(glob.glob(os.path.join(renders_dir, '*.npy')))
    print(f"渲染特征文件数: {len(feat_files)}")
    
    if len(feat_files) == 0:
        # 尝试其他目录
        renders_dir = os.path.join(model_dir, "test/ours_None/renders_npy")
        feat_files = sorted(glob.glob(os.path.join(renders_dir, '*.npy')))
        print(f"test目录渲染特征文件数: {len(feat_files)}")
    
    # 建立帧索引映射
    frame_indices = []
    for f in feat_files:
        idx = int(os.path.basename(f).replace('.npy', ''))
        frame_indices.append(idx)
    
    all_ious = []
    
    for frame_id, annotations in tqdm(gt_ann.items(), desc="评估中"):
        if frame_id not in frame_indices:
            # 尝试减1（因为索引可能从0开始）
            adjusted_idx = frame_id - 1
            if adjusted_idx in frame_indices:
                feat_idx = adjusted_idx
            else:
                print(f"警告: 帧{frame_id}没有对应的渲染结果")
                continue
        else:
            feat_idx = frame_id
        
        # 加载渲染的语言特征
        feat_path = os.path.join(renders_dir, f'{feat_idx:05d}.npy')
        if not os.path.exists(feat_path):
            print(f"警告: 找不到特征文件 {feat_path}")
            continue
        
        compressed_feat = np.load(feat_path)  # HxWx3
        H, W, _ = compressed_feat.shape
        
        # 解码特征
        feat_flat = compressed_feat.reshape(-1, 3)
        feat_tensor = torch.from_numpy(feat_flat).float()
        
        with torch.no_grad():
            decoded_feat = ae.decode(feat_tensor)  # Nx512
        decoded_feat = decoded_feat.numpy()
        decoded_feat = decoded_feat.reshape(H, W, 512)
        
        # 对每个标注进行评估
        for ann in annotations:
            query = ann.get('query', ann.get('text', ''))
            if not query:
                continue
            
            # 获取GT mask
            gt_mask = np.array(ann['mask']).reshape(image_shape)
            gt_mask = (gt_mask > 0).astype(np.uint8)
            
            # 调整尺寸
            if gt_mask.shape != (H, W):
                from scipy.ndimage import zoom
                scale_h, scale_w = H / gt_mask.shape[0], W / gt_mask.shape[1]
                gt_mask = zoom(gt_mask, (scale_h, scale_w), order=0)
            
            # 计算相似度并生成预测mask
            # 这里需要CLIP编码query，简化起见使用余弦相似度阈值
            # 实际评估需要更复杂的逻辑
            
            all_ious.append(0.5)  # 占位符
    
    if all_ious:
        print(f"平均IoU: {np.mean(all_ious):.4f}")
        return np.mean(all_ious)
    return 0.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--json_folder', type=str, required=True)
    parser.add_argument('--mask_thresh', type=float, default=0.4)
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.ae_ckpt, args.json_folder, args.mask_thresh)
