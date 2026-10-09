"""EXP-058 Step A: waldo 全帧 native 分辨率 DINOv2 特征提取 (187 帧)。

native = tile 原尺寸 pad 到 14 倍数, mask 内 patch-mean, 零上采样。
快测已证 native 组 cross top1 59.62% vs CLIP 44.23% (+15.4pp)。
"""
import os, glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys_dir = '/home/xiedexia/project/LangSplat'
import sys
sys.path.insert(0, sys_dir)
sys.path.insert(0, os.path.join(sys_dir, 'eval'))
from mm_langsplat.extractors.dino_extractor import DINOv2Extractor

ROOT = os.path.join(sys_dir, 'dataset/lerf_ovs')
SCENE = 'waldo_kitchen'
SRC = os.path.join(ROOT, SCENE, 'language_features_dino')
DST = os.path.join(ROOT, SCENE, 'language_features_dino_native')
os.makedirs(DST, exist_ok=True)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
ex = DINOv2Extractor(model_name='dinov2_vitb14').cuda().eval()

seg_files = sorted(glob.glob(os.path.join(SRC, '*_s.npy')))
print(f'{len(seg_files)} seg files')
for i, sf in enumerate(seg_files):
    frame = os.path.basename(sf)[:-6]
    out_f = os.path.join(DST, frame + '_f.npy')
    out_s = os.path.join(DST, frame + '_s.npy')
    if os.path.exists(out_f) and os.path.islink(out_s):
        continue
    img = cv2.imread(os.path.join(ROOT, SCENE, 'images', frame + '.jpg'))
    segs = np.load(sf).astype(np.int64)
    ids = np.unique(segs); ids = ids[ids >= 0]
    H, W = segs.shape[1:]
    max_id = int(segs.max())
    nat = np.zeros((max_id + 1, 768), np.float16)
    for t in ids:
        m = (segs == t).any(axis=0)
        ys, xs = np.nonzero(m)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tile = img[y1:y2, x1:x2].copy()
        tile[m[y1:y2, x1:x2] == 0] = 0
        tri = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        h, w, c = tri.shape
        Hn, Wn = ((h + 13) // 14) * 14, ((w + 13) // 14) * 14
        x = torch.zeros(3, Hn, Wn)
        x[:, :h, :w] = torch.from_numpy(tri).permute(2, 0, 1).float() / 255.0
        xn = (x.unsqueeze(0).cuda() - _MEAN) / _STD
        with torch.no_grad():
            o = ex.model.forward_features(xn)['x_norm_patchtokens'][0]
        # mask 内 patch: tile 内容覆盖的网格
        gh, gw = max(1, h // 14), max(1, w // 14)
        sel = torch.zeros(Hn // 14, Wn // 14, dtype=torch.bool)
        sel[:gh, :gw] = True
        nat[t] = F.normalize(o[sel.flatten().cuda()].mean(dim=0), dim=0).half().cpu().numpy()
    np.save(out_f, nat)
    if not os.path.islink(out_s):
        os.symlink(sf, out_s)
    if (i + 1) % 20 == 0:
        print(f'{i+1}/{len(seg_files)}', flush=True)
print('ALL DONE', len(os.listdir(DST)))
