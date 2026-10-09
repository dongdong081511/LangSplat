"""DINOv3 救援快测: 按机理猜测重试 (EXP-057 后续)

假设: DINOv3 tile-mean@224 失败源于 (a) CLS token 未用 (b) 网格密度低 (patch16@224=14×14)
设计: 复用 language_features_dino3 已落盘段表 (免重跑 SAM),
      重建 tile 图像后分别以 224/448 前向, 同时产出 patch-mean 与 cls 两种池化,
      共 4 组特征目录 (m224/c224/m448/c448), 直接复用 tile_retrieval_test.run() 协议。
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

sys.path.insert(0, '/home/xiedexia/project/LangSplat')
sys.path.insert(0, '/home/xiedexia/project/LangSplat/eval')
from mm_langsplat.extractors.dinov3_extractor import DINOv3Extractor, _DINOV3_MEAN, _DINOV3_STD
import torchvision

ROOT = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs'
SCENE = 'waldo_kitchen'
SRC = os.path.join(ROOT, SCENE, 'language_features_dino3')
OUTS = {
    'DINOv3_mean224': os.path.join(ROOT, SCENE, 'dino3_rescue_m224'),
    'DINOv3_cls224': os.path.join(ROOT, SCENE, 'dino3_rescue_c224'),
    'DINOv3_mean448': os.path.join(ROOT, SCENE, 'dino3_rescue_m448'),
    'DINOv3_cls448': os.path.join(ROOT, SCENE, 'dino3_rescue_c448'),
}
_MEAN = torch.tensor(_DINOV3_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(_DINOV3_STD).view(1, 3, 1, 1)


def rebuild_tiles(image_bgr, segs):
    """从 4 级段表并集重建 tile 图像: 任一 level 出现即算该 tile, 特征矩阵按 tile id 对齐。

    返回 (ids, tiles): ids 为全局 tile id 列表, tiles 为对应图像张量列表。
    """
    ids = np.unique(segs)
    ids = ids[ids >= 0]
    tiles = []
    for t in ids:
        m = (segs == t).any(axis=0)
        ys, xs = np.nonzero(m)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tile = image_bgr[y1:y2, x1:x2].copy()
        tile[m[y1:y2, x1:x2] == 0] = 0
        tiles.append(torch.from_numpy(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
    return ids, tiles  # list of [3,h,w]


@torch.no_grad()
def extract(ex, img, segs, res):
    ids, tiles = rebuild_tiles(img, segs)
    batch = torch.stack([F.interpolate(t.unsqueeze(0), size=(res, res), mode='bilinear',
                                       align_corners=False).squeeze(0) for t in tiles]).cuda()
    x = (batch - _MEAN.cuda()) / _STD.cuda()
    feats_mean, feats_cls = [], []
    for i in range(0, len(x), 32):
        out = ex.model.forward_features(x[i:i + 32])
        d = out[0] if isinstance(out, (list, tuple)) else out
        feats_mean.append(F.normalize(d['x_norm_patchtokens'].mean(dim=1), dim=-1).cpu())
        feats_cls.append(F.normalize(d['x_norm_clstoken'], dim=-1).cpu())
    fm = torch.cat(feats_mean).half().numpy()
    fc = torch.cat(feats_cls).half().numpy()
    # 按 tile id 对齐到 (max_id+1, dim) 矩阵, 未出现 id 填 0
    D = fm.shape[1]
    m_full = np.zeros((int(ids.max()) + 1, D), np.float16)
    c_full = np.zeros((int(ids.max()) + 1, D), np.float16)
    m_full[ids] = fm
    c_full[ids] = fc
    return m_full, c_full


def main(limit=None):
    for d in OUTS.values():
        os.makedirs(d, exist_ok=True)
    ex = DINOv3Extractor(verbose=False).cuda()
    seg_files = sorted(glob.glob(os.path.join(SRC, '*_s.npy')))
    n = len(seg_files) if limit is None else min(limit, len(seg_files))
    for i, sf in enumerate(seg_files[:n]):
        frame = os.path.basename(sf)[:-6]  # strip _s.npy
        img = cv2.imread(os.path.join(ROOT, SCENE, 'images', frame + '.jpg'))
        segs = np.load(sf).astype(np.int64)  # [4,H,W] 4 levels
        fm224, fc224 = extract(ex, img, segs, 224)
        fm448, fc448 = extract(ex, img, segs, 448)
        np.save(os.path.join(OUTS['DINOv3_mean224'], frame + '_f.npy'), fm224)
        np.save(os.path.join(OUTS['DINOv3_cls224'], frame + '_f.npy'), fc224)
        np.save(os.path.join(OUTS['DINOv3_mean448'], frame + '_f.npy'), fm448)
        np.save(os.path.join(OUTS['DINOv3_cls448'], frame + '_f.npy'), fc448)
        for d in OUTS.values():  # 段表软链 (与 dino3 同一 _s.npy)
            link = os.path.join(d, frame + '_s.npy')
            if not os.path.islink(link):
                os.symlink(os.path.join(SRC, frame + '_s.npy'), link)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f'{i+1}/{n} frames done', flush=True)
    # 复用快测协议
    import tile_retrieval_test as trt
    gt = trt.load_gt()
    for tag, d in OUTS.items():
        trt.run(d, gt, tag)


if __name__ == '__main__':
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
