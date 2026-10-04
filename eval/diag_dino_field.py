"""Diagnose figurines DINO field chain: AE roundtrip vs 32d-encoded retrieval."""
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'autoencoder'))
from model import Autoencoder

device = 'cuda'

def encode(ae, x):
    for m in ae.encoder: x = m(x)
    return x

def decode(ae, x):
    for m in ae.decoder: x = m(x)
    return x

def load_ae(name):
    ae = Autoencoder([256,128,32,32,32], [32,128,256,256,768], input_dim=768).to(device)
    ae.load_state_dict(torch.load(f'{ROOT}/ckpt/{name}/best_ckpt.pth', map_location=device))
    ae.eval()
    return ae

def roundtrip_cos(ae, scene, n=60):
    feats = sorted(glob.glob(f'{ROOT}/dataset/lerf_ovs/{scene}/language_features_dino/*_f.npy'))[:n]
    cos = []
    with torch.no_grad():
        for f in feats:
            x = torch.from_numpy(np.load(f)).float().to(device)  # [T,768]
            z = encode(ae, x); xr = decode(ae, z)
            cos.append(F.cosine_similarity(x, xr, dim=-1).mean().item())
    return np.mean(cos), np.std(cos)

if __name__ == '__main__':
    for scene in ['teatime', 'figurines']:
        ae = load_ae(f'{scene}_dino_32d')
        m, s = roundtrip_cos(ae, scene)
        print(f'[AE roundtrip] {scene} dino 32d: cos mean={m:.4f} std={s:.4f} (n=60 frames)')
    # rank agreement: 768d neighbors vs 32d neighbors (frame 0, all tiles)
    for scene in ['teatime', 'figurines']:
        d768 = f'{ROOT}/dataset/lerf_ovs/{scene}/language_features_dino'
        d32 = f'{ROOT}/dataset/lerf_ovs/{scene}/language_features_dim32_dino'
        f0 = sorted(glob.glob(d768+'/*_f.npy'))[0]
        b0 = os.path.basename(f0).replace('_f.npy','')
        x = torch.from_numpy(np.load(f0)).float()          # [T,768]
        z = torch.from_numpy(np.load(f'{d32}/{b0}_f.npy')).float()  # [T,32]
        S768 = (x @ x.T); S32 = (z @ z.T)
        T = min(S768.shape[0], 300)
        # top-1 neighbor agreement
        n768 = S768[:T,:T].fill_diagonal_(-9).argmax(dim=1)
        n32 = S32[:T,:T].fill_diagonal_(-9).argmax(dim=1)
        agree = (n768 == n32).float().mean().item()
        # sim distribution spread
        print(f'[rank agree] {scene}: top1-neighbor 768d vs 32d agree={agree:.3f}; '
              f'sim std 768d={S768.std():.4f} 32d={S32.std():.4f}')
