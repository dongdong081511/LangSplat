#!/usr/bin/env python
"""LangSplat Text Query Script"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autoencoder'))

import open_clip
import torchvision
from autoencoder.model import Autoencoder


class LangSplatQuery:
    def __init__(self, model_path, ae_ckpt_path, encoder_dims, decoder_dims, device='cuda'):
        self.device = device
        self.model_path = Path(model_path)
        
        print("Loading autoencoder...")
        self.autoencoder = Autoencoder(encoder_dims, decoder_dims).to(device)
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        self.autoencoder.load_state_dict(checkpoint)
        self.autoencoder.eval()
        
        print("Loading CLIP model...")
        self.clip_model_type = "ViT-B-16"
        self.clip_pretrained = 'laion2b_s34b_b88k'
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type, pretrained=self.clip_pretrained, precision="fp16",
        )
        self.model.eval()
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        
        self.negatives = ("object", "things", "stuff", "texture")
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = self.model.encode_text(tok_phrases)
            self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
        
        self.pos_embeds = None
        print("Model loaded!")
    
    def encode_text(self, text_list):
        with torch.no_grad():
            text = self.tokenizer(text_list).to(self.device)
            text_embeds = self.model.encode_text(text)
            text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds
    
    def set_query(self, query_text):
        if isinstance(query_text, str):
            query_text = [query_text]
        self.queries = query_text
        self.pos_embeds = self.encode_text(query_text)
        print(f"Query: {query_text}")
    
    def decode_features(self, low_dim_features):
        with torch.no_grad():
            features = low_dim_features.unsqueeze(0)
            decoded = self.autoencoder.decode(features)
            decoded = decoded.squeeze(0)
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
    
    def query_image(self, npy_path, output_path=None, threshold=0.5):
        low_dim = np.load(npy_path)
        low_dim = torch.from_numpy(low_dim).float().to(self.device)
        high_dim = self.decode_features(low_dim)
        h, w, c = high_dim.shape
        flat_features = high_dim.view(-1, c)
        
        results = []
        for i, query in enumerate(self.queries):
            relevancy = self.get_relevancy(flat_features, i)
            pos_prob = relevancy[:, 0].view(h, w)
            results.append(pos_prob)
        
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for i, (query, result) in enumerate(zip(self.queries, results)):
                heatmap = (result.cpu().numpy() * 255).astype(np.uint8)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(str(output_path / f'{query.replace(" ", "_")}_heatmap.png'), heatmap_colored)
                mask = ((result.cpu().numpy() > threshold) * 255).astype(np.uint8)
                cv2.imwrite(str(output_path / f'{query.replace(" ", "_")}_mask.png'), mask)
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--output', type=str, default='query_results')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--encoder_dims', type=int, nargs='+', default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', type=int, nargs='+', default=[16, 32, 64, 128, 256, 256, 512])
    parser.add_argument('--image_idx', type=int, default=None)
    args = parser.parse_args()
    
    q = LangSplatQuery(args.model_path, args.ae_ckpt, args.encoder_dims, args.decoder_dims)
    q.set_query([x.strip() for x in args.query.split(',')])
    
    npy_path = Path(args.model_path) / 'train' / 'ours_None' / 'renders_npy'
    npy_files = sorted(npy_path.glob('*.npy'))
    print(f"Found {len(npy_files)} files")
    
    if args.image_idx is not None:
        npy_files = [npy_files[args.image_idx]]
    
    for f in tqdm(npy_files):
        q.query_image(f, Path(args.output) / f.stem, args.threshold)
    print(f"Done! Results in {args.output}")
