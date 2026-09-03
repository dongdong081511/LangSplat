"""
LangSplat 3D-OVS 评估脚本
评估sofa场景的mIoU
"""
import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import open_clip

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
    """加载GT标注，返回 {frame_id: [annotations]}"""
    gt_ann = {}
    json_files = sorted(glob.glob(os.path.join(json_folder, 'frame_*.json')))
    
    for jf in json_files:
        # 从文件名提取frame_id (frame_00003.json -> 3)
        frame_name = os.path.basename(jf)
        frame_id = int(frame_name.replace('frame_', '').replace('.json', ''))
        
        with open(jf) as f:
            data = json.load(f)
        
        # 存储所有对象
        gt_ann[frame_id] = {
            'info': data.get('info', {}),
            'objects': data.get('objects', [])
        }
    
    return gt_ann


def polygon_to_mask(polygon, height, width):
    """将多边形转换为二进制mask"""
    from PIL import Image, ImageDraw
    img = Image.new('L', (width, height), 0)
    if len(polygon) > 2:
        # polygon是[x,y]列表
        flat_polygon = [(p[0], p[1]) for p in polygon]
        ImageDraw.Draw(img).polygon(flat_polygon, fill=1)
    return np.array(img, dtype=np.uint8)


def compute_iou(pred_mask, gt_mask):
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_model(model_dir, ae_ckpt_path, json_folder, clip_model_name='ViT-B-16', clip_pretrained='laion2b_s34b_b88k'):
    """评估单个模型的mIoU"""
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
    print(f"  Autoencoder加载完成: {ae_ckpt_path}")
    
    # ========== 加载CLIP ==========
    print("\n[2/4] 加载CLIP模型...")
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    print(f"  CLIP模型: {clip_model_name} ({clip_pretrained})")
    
    # ========== 加载GT标注 ==========
    print("\n[3/4] 加载GT标注...")
    gt_annotations = load_gt_annotations(json_folder)
    print(f"  GT帧数: {len(gt_annotations)}")
    
    # ========== 查找渲染结果 ==========
    renders_dir = os.path.join(model_dir, "train/ours_None/renders_npy")
    feat_files = sorted(glob.glob(os.path.join(renders_dir, '*.npy')))
    print(f"  渲染特征文件数: {len(feat_files)}")
    
    if len(feat_files) == 0:
        print("错误: 未找到渲染结果!")
        return 0.0
    
    # 建立帧索引映射 (00000.npy -> 0, 00001.npy -> 1, ...)
    # GT中 frame_00003 -> 3 需要映射到正确的渲染文件
    render_indices = {}
    for f in feat_files:
        idx = int(os.path.basename(f).replace('.npy', ''))
        render_indices[idx] = f
    
    # ========== 评估 ==========
    print("\n[4/4] 开始评估...")
    
    all_results = []  # 存储每个query的结果
    category_results = {}  # 按类别统计
    
    for frame_id, ann_data in tqdm(gt_annotations.items(), desc="评估帧"):
        # 查找对应的渲染文件
        # GT: frame_00003 -> frame_id=3
        # Render: 00000.npy 对应第一张图
        # 需要确定映射关系
        
        # 方案: frame_id可能是图片索引(从0开始)
        if frame_id in render_indices:
            feat_path = render_indices[frame_id]
        else:
            # 尝试其他映射
            continue
        
        # 加载渲染的语言特征
        compressed_feat = np.load(feat_path)  # HxWx3
        H, W, _ = compressed_feat.shape
        
        # 解码特征
        feat_flat = compressed_feat.reshape(-1, 3)
        feat_tensor = torch.from_numpy(feat_flat).float().to(device)
        
        with torch.no_grad():
            decoded_feat = ae.decode(feat_tensor)  # Nx512
        decoded_feat = decoded_feat.cpu().numpy()
        decoded_feat = decoded_feat.reshape(H, W, 512)
        
        # 归一化
        feat_norm = decoded_feat / (np.linalg.norm(decoded_feat, axis=-1, keepdims=True) + 1e-8)
        
        # 获取图像尺寸
        info = ann_data.get('info', {})
        gt_height = info.get('height', 1080)
        gt_width = info.get('width', 1440)
        
        # 评估每个对象
        for obj in ann_data.get('objects', []):
            category = obj.get('category', '')
            query = category  # 使用类别名作为query
            
            if not query:
                continue
            
            # 获取GT mask
            segmentation = obj.get('segmentation', [])
            if not segmentation:
                continue
            
            gt_mask = polygon_to_mask(segmentation, gt_height, gt_width)
            
            # 调整GT mask尺寸以匹配渲染结果
            if gt_mask.shape != (H, W):
                from scipy.ndimage import zoom
                scale_h, scale_w = H / gt_mask.shape[0], W / gt_mask.shape[1]
                gt_mask = zoom(gt_mask, (scale_h, scale_w), order=0)
            
            # 编码query
            text_tokens = tokenizer([query]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_feat = text_features.cpu().numpy()[0]  # 512
            
            # 计算相似度
            similarity = np.dot(feat_norm, text_feat)  # HxW
            
            # 生成预测mask (阈值法)
            pred_mask = (similarity > 0.3).astype(np.uint8)
            
            # 计算IoU
            iou = compute_iou(pred_mask, gt_mask)
            
            all_results.append({
                'frame_id': frame_id,
                'category': category,
                'iou': iou
            })
            
            # 按类别统计
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(iou)
    
    # ========== 输出结果 ==========
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    
    if len(all_results) == 0:
        print("警告: 没有成功评估任何样本!")
        print("可能原因: 帧索引映射不正确")
        return 0.0
    
    # 总体mIoU
    all_ious = [r['iou'] for r in all_results]
    miou = np.mean(all_ious)
    print(f"\n总体mIoU: {miou:.4f}")
    print(f"评估样本数: {len(all_ious)}")
    
    # 按类别输出
    print(f"\n按类别IoU:")
    print(f"{'类别':<30} {'IoU':>8} {'样本数':>8}")
    print("-" * 50)
    for cat, ious in sorted(category_results.items()):
        cat_miou = np.mean(ious)
        print(f"{cat:<30} {cat_miou:>8.4f} {len(ious):>8}")
    
    return miou


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LangSplat 3D-OVS评估')
    parser.add_argument('--model_dir', type=str, default='output/sofa_1',
                        help='模型目录')
    parser.add_argument('--ae_ckpt', type=str, default='autoencoder/ckpt/sofa/ae_ckpt/best_ckpt.pth',
                        help='Autoencoder checkpoint路径')
    parser.add_argument('--json_folder', type=str, default='output/sofa_data/segmentations_json_v3',
                        help='GT标注JSON文件夹')
    parser.add_argument('--clip_model', type=str, default='ViT-B-16',
                        help='CLIP模型名称')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_s34b_b88k',
                        help='CLIP预训练权重')
    args = parser.parse_args()
    
    evaluate_model(
        args.model_dir, 
        args.ae_ckpt, 
        args.json_folder,
        args.clip_model,
        args.clip_pretrained
    )
