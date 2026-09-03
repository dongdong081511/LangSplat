#!/usr/bin/env python
"""LangSplat 文本查询测试脚本"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.append('.')
sys.path.append('./eval')
sys.path.append('./autoencoder')

import open_clip
import torchvision
from autoencoder.model import Autoencoder


class LangSplatQuery:
    def __init__(self, ae_ckpt_path, encoder_dims, decoder_dims, device='cuda'):
        self.device = device
        
        # 加载自编码器
        print("Loading autoencoder...")
        self.autoencoder = Autoencoder(encoder_dims, decoder_dims).to(device)
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        self.autoencoder.load_state_dict(checkpoint)
        self.autoencoder.eval()
        
        # 加载CLIP模型
        print("Loading CLIP model...")
        self.clip_model_type = "ViT-B-16"
        self.clip_pretrained = 'laion2b_s34b_b88k'
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type, pretrained=self.clip_pretrained, precision="fp16",
        )
        self.model.eval()
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        
        # 默认负样本
        self.negatives = ("object", "things", "stuff", "texture")
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(p) for p in self.negatives]).to(device)
            self.neg_embeds = self.model.encode_text(tok_phrases)
            self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        
        self.pos_embeds = None
        self.queries = None
        print("Model loaded!")
    
    def set_query(self, query_text):
        if isinstance(query_text, str):
            query_text = [query_text]
        self.queries = query_text
        with torch.no_grad():
            tokens = torch.cat([self.tokenizer(q) for q in query_text]).to(self.device)
            self.pos_embeds = self.model.encode_text(tokens)
            self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        print(f"Query set: {query_text}")
    
    def decode_features(self, low_dim):
        with torch.no_grad():
            low_dim = torch.from_numpy(low_dim).float().to(self.device)
            decoded = self.autoencoder.decode(low_dim)
        return decoded
    
    def get_relevancy(self, embed, positive_id=0):
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        
        positive_vals = output[..., positive_id:positive_id+1]
        negative_vals = output[..., len(self.queries):]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))
        
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        result = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]
        return result
    
    def query_image(self, npy_path, output_dir, threshold=0.5):
        # 加载特征
        low_dim = np.load(npy_path)
        high_dim = self.decode_features(low_dim)
        
        h, w, c = high_dim.shape
        flat = high_dim.view(-1, c)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载对应的RGB图像
        rgb_path = str(npy_path).replace('renders_npy', 'renders').replace('.npy', '.png')
        if os.path.exists(rgb_path):
            rgb = cv2.imread(rgb_path)
        else:
            rgb = None
        
        for i, query in enumerate(self.queries):
            relevancy = self.get_relevancy(flat, i)
            score = relevancy[:, 0].view(h, w).cpu().numpy()
            
            # 热力图
            heatmap = (score * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(f"{output_dir}/{query.replace(' ', '_')}_heatmap.png", heatmap_color)
            
            # Mask
            mask = ((score > threshold) * 255).astype(np.uint8)
            cv2.imwrite(f"{output_dir}/{query.replace(' ', '_')}_mask.png", mask)
            
            # 叠加结果
            if rgb is not None:
                overlay = rgb.copy()
                mask_bool = score > threshold
                overlay[mask_bool] = (0.7 * overlay[mask_bool] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.imwrite(f"{output_dir}/{query.replace(' ', '_')}_overlay.png", overlay)
            
            print(f"  Saved: {query}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='output/lerf_figurines_langsplat_1')
    parser.add_argument('--ae_ckpt', type=str, default='autoencoder/ckpt/lerf_figurines/best_ckpt.pth')
    parser.add_argument('--query', type=str, default='figurine')
    parser.add_argument('--output', type=str, default='query_results')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--encoder_dims', type=int, nargs='+', default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', type=int, nargs='+', default=[16, 32, 64, 128, 256, 256, 512])
    parser.add_argument('--image_idx', type=int, default=0, help='Image index to query (-1 for all)')
    args = parser.parse_args()
    
    # 初始化
    q = LangSplatQuery(args.ae_ckpt, args.encoder_dims, args.decoder_dims)
    q.set_query([x.strip() for x in args.query.split(',')])
    
    # 查找特征文件
    npy_dir = Path(args.model_path) / 'train' / 'ours_None' / 'renders_npy'
    npy_files = sorted(npy_dir.glob('*.npy'))
    print(f"Found {len(npy_files)} images")
    
    if args.image_idx >= 0:
        npy_files = [npy_files[args.image_idx]]
    
    # 执行查询
    output_dir = Path(args.output)
    for f in tqdm(npy_files, desc="Querying"):
        q.query_image(f, output_dir / f.stem, args.threshold)
    
    print(f"\nDone! Results saved to: {args.output}")
