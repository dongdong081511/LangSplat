# multires_pyramid.py
"""方案A: 图像金字塔多分辨率 SAM 分割 (SAM 之前)

对输入图像生成高斯金字塔 (1.0x, 0.5x, ...), 在每个分辨率上分别运行 SAM,
将 mask 上采样到原始分辨率后做跨尺度 NMS 合并。

核心思想: 不同分辨率下 SAM 关注的物体尺度不同:
  - 高分辨率: 捕获细节和小物体 (茶杯把手)
  - 低分辨率: 捕获大结构和整体区域 (整张桌子)

数学原理:
  - 高斯金字塔: G_{l+1} = Downsample(GaussianBlur(G_l))
  - 上采样: mask 在低分辨率生成, 用最近邻插值回原始尺寸
  - 跨尺度 NMS: 对不同尺度的 mask 计算 IoU, 保留得分更高者
"""
import cv2
import numpy as np
import torch

# 全局模式
MODE = "none"
LEVELS = 2
CROSS_IOU_THR = 0.7


def set_mode(mode: str = "none", levels: int = 2):
    global MODE, LEVELS
    MODE = mode
    LEVELS = levels


def generate_pyramid(image_np, levels=2):
    """生成高斯金字塔。levels=2 → [1.0x, 0.5x]; levels=3 → [1.0x, 0.5x, 0.25x]"""
    pyramid = [image_np]
    current = image_np
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _upsample_masks(masks, orig_h, orig_w):
    """将 mask 的 segmentation 上采样到原始分辨率"""
    for m in masks:
        seg = m['segmentation']
        seg_up = cv2.resize(
            seg.astype(np.uint8), (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        m['segmentation'] = seg_up
        ys, xs = np.where(seg_up)
        if len(ys) > 0:
            m['bbox'] = [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1),
                         int(ys.max() - ys.min() + 1)]


def _cross_scale_nms(masks, iou_thr=0.7, score_thr=0.1, inner_thr=0.2):
    """跨尺度 NMS: 去除不同尺度间的重复 mask"""
    if len(masks) == 0:
        return []
    # 延迟导入, 避免循环依赖
    from preprocess import mask_nms, filter
    seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks], axis=0))
    iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks], axis=0))
    stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks], axis=0))
    scores = stability * iou_pred
    keep_idx = mask_nms(seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)
    return filter(keep_idx, masks)


def run_sam_multiscale(mask_generator, image_np, levels=2, cross_iou_thr=0.7):
    """在多分辨率上运行 SAM 并合并 mask

    Args:
        mask_generator: SamAutomaticMaskGenerator 实例
        image_np: RGB numpy 图像 (H, W, 3)
        levels: 金字塔层数
        cross_iou_thr: 跨尺度 NMS IoU 阈值

    Returns:
        (masks_default, masks_s, masks_m, masks_l) 合并后的 4 级 mask
    """
    from preprocess import masks_update

    pyramid = generate_pyramid(image_np, levels)
    orig_h, orig_w = image_np.shape[:2]

    all_masks = {'default': [], 's': [], 'm': [], 'l': []}

    for img_scaled in pyramid:
        # 在当前分辨率运行 SAM
        masks_d, masks_s, masks_m, masks_l = mask_generator.generate(img_scaled)
        # 尺度内 NMS (与原 pipeline 一致)
        masks_d, masks_s, masks_m, masks_l = masks_update(
            masks_d, masks_s, masks_m, masks_l,
            iou_thr=0.8, score_thr=0.7, inner_thr=0.5
        )
        # 上采样 mask 到原始分辨率
        for mask_list in [masks_d, masks_s, masks_m, masks_l]:
            _upsample_masks(mask_list, orig_h, orig_w)

        all_masks['default'].extend(masks_d)
        all_masks['s'].extend(masks_s)
        all_masks['m'].extend(masks_m)
        all_masks['l'].extend(masks_l)

    # 跨尺度 NMS, 每级独立合并
    merged = {}
    for mode in ['default', 's', 'm', 'l']:
        merged[mode] = _cross_scale_nms(all_masks[mode], iou_thr=cross_iou_thr)

    return merged['default'], merged['s'], merged['m'], merged['l']
