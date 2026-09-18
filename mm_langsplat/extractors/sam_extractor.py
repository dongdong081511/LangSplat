"""SAM (Segment Anything) 三层级 mask 提取器。

复用 LangSplat preprocess.py 中的 SAM 逻辑, 生成 4 级 mask:
default, subpart (s), part (m), whole (l)。

Reference:
    Kirillov et al., "Segment Anything", ICCV 2023.
    https://arxiv.org/abs/2304.02643
"""
import numpy as np
import torch
import torch.nn as nn
import cv2

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    raise ImportError(
        "segment_anything 未安装。"
        "请运行: pip install git+https://github.com/facebookresearch/segment-anything.git"
    )


class SAMExtractor(nn.Module):
    """SAM 自动 mask 生成器, 输出多层级的 tile 图像和索引图。

    Args:
        ckpt_path: SAM ViT-H 权重路径
        points_per_side: 每边网格点数
        pred_iou_thresh: 预测 IoU 阈值
        box_nms_thresh: box NMS 阈值
    """

    def __init__(self,
                 ckpt_path='ckpts/sam_vit_h_4b8939.pth',
                 points_per_side=32,
                 pred_iou_thresh=0.7,
                 box_nms_thresh=0.7,
                 stability_score_thresh=0.85,
                 crop_n_layers=1,
                 crop_n_points_downscale_factor=1,
                 min_mask_region_area=100):
        super().__init__()
        sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            box_nms_thresh=box_nms_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )

    @torch.no_grad()
    def forward(self, image):
        """为图像生成 SAM mask。

        Args:
            image: [B, 3, H, W], 范围 [0, 1] (uint8 兼容)
        Returns:
            seg_images: dict {mode: [N, 3, 224, 224]} mask 后的 padded tile
            seg_maps:   dict {mode: [H, W] int32} 每像素的 tile 索引
        """
        import os
        import sys

        # 将 preprocess.py 所在目录加入 sys.path 以复用 NMS 逻辑
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from preprocess import masks_update

        img_np = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(img_np)
        masks_default, masks_s, masks_m, masks_l = masks_update(
            masks_default, masks_s, masks_m, masks_l,
            iou_thr=0.8, score_thr=0.7, inner_thr=0.5
        )

        seg_images, seg_maps = {}, {}
        seg_images['default'], seg_maps['default'] = self._mask2segmap(masks_default, img_np)
        if len(masks_s) != 0:
            seg_images['s'], seg_maps['s'] = self._mask2segmap(masks_s, img_np)
        if len(masks_m) != 0:
            seg_images['m'], seg_maps['m'] = self._mask2segmap(masks_m, img_np)
        if len(masks_l) != 0:
            seg_images['l'], seg_maps['l'] = self._mask2segmap(masks_l, img_np)

        return seg_images, seg_maps

    @staticmethod
    def _mask2segmap(masks, image):
        """将 SAM mask 转换为 padded tile + 索引图。"""
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i, mask in enumerate(masks):
            seg_img = SAMExtractor._get_seg_img(mask, image)
            pad_seg_img = cv2.resize(SAMExtractor._pad_img(seg_img), (224, 224))
            seg_img_list.append(pad_seg_img)
            seg_map[mask['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0)
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0)
        return seg_imgs, seg_map

    @staticmethod
    def _get_seg_img(mask, image):
        """从图像中裁剪 mask 区域。"""
        image = image.copy()
        image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
        x, y, w, h = np.int32(mask['bbox'])
        return image[y:y + h, x:x + w, ...]

    @staticmethod
    def _pad_img(img):
        """将图像 pad 到正方形。"""
        h, w, _ = img.shape
        l = max(w, h)
        pad = np.zeros((l, l, 3), dtype=np.uint8)
        if h > w:
            pad[:, (h - w) // 2:(h - w) // 2 + w, :] = img
        else:
            pad[(w - h) // 2:(w - h) // 2 + h, :, :] = img
        return pad
