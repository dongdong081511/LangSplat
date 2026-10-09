"""EXP-058: waldo DINOv2 入口优势救援 — 6 组池化/分辨率消融 + tile 纯度诊断。

背景: waldo 是任务路由唯一负场, 归因=DINO 2D 入口优势归零 (42.3% vs CLIP 44.2%)。
EXP-057d 只对 DINOv3 测过 masked-mean (坍缩救不动); DINOv2 特征空间健康 (neg cos 0.441),
黑 patch 稀释/分辨率/上下文这些旋钮可能真有效。

组别 (全部复用 language_features_dino 段表, 4 级并集重建 tile):
  base224 : full patch-mean @224 (基线复现, 应≈42.31%)
  m224    : masked patch-mean @224 (剔除黑 patch)
  f448    : full patch-mean @448 (32×32 网格, 小物体)
  m448    : masked patch-mean @448
  pad10   : bbox 外扩 10% 上下文 + full mean @224
  gem3    : generalized mean p=3 @224 (强调显著 patch)
诊断: tile 尺寸分布 + tile 实例纯度 (一个 tile 跨多个 GT 实例的比例)。
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, '/home/xiedexia/project/LangSplat')
sys.path.insert(0, '/home/xiedexia/project/LangSplat/eval')
from tile_retrieval_test import load_gt
from mm_langsplat.extractors.dino_extractor import DINOv2Extractor

ROOT = '/home/xiedexia/project/LangSplat/dataset/lerf_ovs'
SCENE = 'waldo_kitchen'
SRC = os.path.join(ROOT, SCENE, 'language_features_dino')
OUTS = ['dino2_base224', 'dino2_m224', 'dino2_f448', 'dino2_m448', 'dino2_pad10', 'dino2_gem3', 'dino2_native']
for d in OUTS:
    os.makedirs(os.path.join(ROOT, SCENE, d), exist_ok=True)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

ex = DINOv2Extractor(model_name='dinov2_vitb14').cuda().eval()
gt = load_gt()
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else len(gt)


def build_tiles(img_bgr, segs, pad=0.0):
    """4 级并集 tile, pad 为 bbox 外扩比例。返回 (ids, tiles[3,h,w], masks[bool h,w])"""
    ids = np.unique(segs); ids = ids[ids >= 0]
    H, W = segs.shape[1:]
    tiles, masks = [], []
    for t in ids:
        m = (segs == t).any(axis=0)
        ys, xs = np.nonzero(m)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ph, pw = int((y2 - y1) * pad), int((x2 - x1) * pad)
        Y1, Y2 = max(0, y1 - ph), min(H, y2 + ph)
        X1, X2 = max(0, x1 - pw), min(W, x2 + pw)
        tile = img_bgr[Y1:Y2, X1:X2].copy()
        tile[m[Y1:Y2, X1:X2] == 0] = 0
        tiles.append(torch.from_numpy(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0)
        masks.append(torch.from_numpy(m[Y1:Y2, X1:X2]))
    return ids, tiles, masks


def fwd(tiles, res):
    x = torch.stack([F.interpolate(t.unsqueeze(0), size=(res, res), mode='bilinear',
                                   align_corners=False).squeeze(0) for t in tiles]).cuda()
    x = (x - _MEAN) / _STD
    with torch.no_grad():
        out = ex.model.forward_features(x)
        return out['x_norm_patchtokens']  # [N, G*G, D]


def to_full(ids, feats, max_id):
    M = np.zeros((max_id + 1, feats.shape[1]), np.float16)
    M[ids] = F.normalize(feats, dim=-1).half().cpu().numpy()
    return M


# ---- 诊断累加器 ----
tile_sizes, purity = [], []
n_done = 0
for frame, objs in list(gt.items())[:N_FRAMES]:
    sp = os.path.join(SRC, frame + '_s.npy')
    fp = os.path.join(SRC, frame + '_f.npy')
    if not (os.path.exists(sp) and os.path.exists(fp)):
        continue
    img = cv2.imread(os.path.join(ROOT, SCENE, 'images', frame + '.jpg'))
    segs = np.load(sp).astype(np.int64)
    ids, tiles, masks = build_tiles(img, segs)
    max_id = int(segs.max())
    if len(tiles) == 0:
        continue

    # 诊断: tile 尺寸 + 纯度 (tile 内 GT 实例数)
    for n, t in enumerate(ids):
        m = (segs == t).any(axis=0)
        tile_sizes.append(int(m.sum()))
        n_inst = sum(1 for o in objs if (o['mask'] & m).sum() > 0.1 * min(o['mask'].sum(), m.sum()))
        purity.append(n_inst)
    # 基线复现直接读原特征 (同协议 full-mean@224)
    f0 = np.load(fp).astype(np.float32)
    np.save(os.path.join(ROOT, SCENE, 'dino2_base224', frame + '_f.npy'),
            f0 / (np.linalg.norm(f0, axis=1, keepdims=True) + 1e-8))

    # 224 前向 -> m224 / gem3
    pt224 = fwd(tiles, 224)  # [N, 256, D]
    G = 16
    mm, gm = [], []
    for n in range(len(tiles)):
        r = F.interpolate(tiles[n].unsqueeze(0), size=(224, 224), mode='bilinear',
                          align_corners=False).squeeze(0)
        # 14×14 patch 网格的黑判定: mask resize 到网格
        gmask = F.adaptive_max_pool2d(masks[n].unsqueeze(0).float(), (G, G)).squeeze(0).bool().flatten()
        if gmask.sum() == 0:
            gmask[:] = True
        mm.append(F.normalize(pt224[n][gmask].mean(dim=0), dim=0))
        gm.append(F.normalize((pt224[n] ** 3).mean(dim=0) ** (1 / 3), dim=0))  # geM p=3 全 patch
    np.save(os.path.join(ROOT, SCENE, 'dino2_m224', frame + '_f.npy'), to_full(ids, torch.stack(mm), max_id))
    np.save(os.path.join(ROOT, SCENE, 'dino2_gem3', frame + '_f.npy'), to_full(ids, torch.stack(gm), max_id))

    # 448 前向 -> f448 / m448
    pt448 = fwd(tiles, 448)
    G4 = 32
    mm4 = []
    for n in range(len(tiles)):
        gmask = F.adaptive_max_pool2d(masks[n].unsqueeze(0).float(), (G4, G4)).squeeze(0).bool().flatten()
        if gmask.sum() == 0:
            gmask[:] = True
        mm4.append(F.normalize(pt448[n][gmask].mean(dim=0), dim=0))
    np.save(os.path.join(ROOT, SCENE, 'dino2_f448', frame + '_f.npy'),
            to_full(ids, F.normalize(pt448.mean(dim=1), dim=-1), max_id))
    np.save(os.path.join(ROOT, SCENE, 'dino2_m448', frame + '_f.npy'), to_full(ids, torch.stack(mm4), max_id))

    # pad10 -> full mean 224
    _, tiles_p, _ = build_tiles(img, segs, pad=0.10)
    ptp = fwd(tiles_p, 224)
    np.save(os.path.join(ROOT, SCENE, 'dino2_pad10', frame + '_f.npy'),
            to_full(ids, F.normalize(ptp.mean(dim=1), dim=-1), max_id))

    # native: 原分辨率 (pad 到 14 倍数, 零上采样), mask 内 patch-mean
    nat = []
    for n in range(len(tiles)):
        c, h, w = tiles[n].shape
        Hn, Wn = ((h + 13) // 14) * 14, ((w + 13) // 14) * 14
        x = torch.zeros(3, Hn, Wn)
        x[:, :h, :w] = tiles[n]
        gmask = torch.zeros(Hn // 14, Wn // 14, dtype=torch.bool)
        gm_h, gm_w = min(h, Hn), min(w, Wn)
        gmask[:max(1, gm_h // 14), :max(1, gm_w // 14)] = True  # tile 实际内容覆盖的 patch
        xn = ((x.unsqueeze(0).cuda() - _MEAN) / _STD)
        with torch.no_grad():
            o = ex.model.forward_features(xn)['x_norm_patchtokens'][0]  # [P, D]
        sel = gmask.flatten().cuda()
        if sel.sum() == 0:
            sel[:] = True
        nat.append(F.normalize(o[sel].mean(dim=0), dim=0))
    np.save(os.path.join(ROOT, SCENE, 'dino2_native', frame + '_f.npy'),
            to_full(ids, torch.stack(nat), max_id))

    for d in OUTS:
        link = os.path.join(ROOT, SCENE, d, frame + '_s.npy')
        if not os.path.islink(link):
            os.symlink(sp, link)
    n_done += 1
    if n_done % 20 == 0:
        print(f'{n_done}/{N_FRAMES} frames', flush=True)

ts, pu = np.array(tile_sizes), np.array(purity)
print(f'\n[诊断] tiles={len(ts)}  面积 px: median={np.median(ts):.0f} p10={np.percentile(ts,10):.0f} '
      f'p90={np.percentile(ts,90):.0f}  小tile(<2000px)占比={float((ts<2000).mean()):.1%}')
print(f'[诊断] tile 跨 GT 实例数: 1个={float((pu==1).mean()):.1%}  2个={float((pu==2).mean()):.1%}  '
      f'>=3个={float((pu>=3).mean()):.1%}')

# ---- 快测协议 ----
import tile_retrieval_test as trt
gt2 = trt.load_gt()
for tag in OUTS:
    trt.run(os.path.join(ROOT, SCENE, tag), gt2, tag)
