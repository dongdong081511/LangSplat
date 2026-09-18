"""多模态离线特征提取脚本。

对场景中所有图像提取 CLIP、DINOv2、Depth、Normal、Texture 特征,
并按模态分目录保存 per-frame .pt 文件。

输出目录结构:
  <output_dir>/
  ├── clip/       frame_00001.pt   [512, H, W]
  ├── dino/       frame_00001.pt   [768, H, W]
  ├── depth/      frame_00001.pt   [1, H, W]
  ├── normal/     frame_00001.pt   [3, H, W]
  └── texture/    frame_00001.pt   [64, H, W]

用法:
    python -m mm_langsplat.preprocess_mm \
        --scene_dir data/LERF/teatime \
        --output_dir data/LERF/teatime/features_mm
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm_langsplat.extractors import MultiModalExtractor


def load_image(image_path, resolution=-1):
    """加载图像为 [1, 3, H, W] float 张量, 范围 [0, 1]。

    镜像 LangSplat preprocess.py 的图像加载逻辑 (含可选降分辨率)。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    orig_h, orig_w = image.shape[:2]

    if resolution == -1:
        if orig_h > 1080:
            global_down = orig_h / 1080.0
        else:
            global_down = 1.0
    else:
        global_down = orig_w / float(resolution)

    scale = float(global_down)
    new_size = (int(orig_w / scale), int(orig_h / scale))
    image = cv2.resize(image, new_size)

    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image.unsqueeze(0)  # [1, 3, H, W]


def preprocess_scene(scene_dir, output_dir, device='cuda',
                     use_clip=True, use_dino=True,
                     use_depth=False, use_normal=False, use_texture=False,
                     resolution=-1, dino_model_name='dinov2_vitb14'):
    """对场景中所有图像提取多模态特征。

    Args:
        scene_dir: 场景目录路径 (需含 'images/' 子目录)
        output_dir: 输出根目录 (内部按模态创建子目录)
        device: 'cuda' 或 'cpu'
        use_clip/dino/depth/normal/texture: 各模态启用开关
        resolution: 最大宽度 (-1 = 自动对 >1080p 降分辨率)
        dino_model_name: DINOv2 变体名
    """
    extractor = MultiModalExtractor(
        use_clip=use_clip,
        use_dino=use_dino,
        use_depth=use_depth,
        use_normal=use_normal,
        use_texture=use_texture,
        dino_model_name=dino_model_name,
    ).to(device)
    print(f"已启用模态: {extractor.enabled_modalities}")
    print(f"输出维度: {extractor.dims}")

    image_dir = os.path.join(scene_dir, 'images')
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图像目录未找到: {image_dir}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"在 {image_dir} 中找到 {len(image_files)} 张图像")

    # 创建输出子目录
    for modality in extractor.enabled_modalities:
        os.makedirs(os.path.join(output_dir, modality), exist_ok=True)

    for img_file in tqdm(image_files, desc=f"处理 {os.path.basename(scene_dir)}"):
        image_path = os.path.join(image_dir, img_file)
        image = load_image(image_path, resolution=resolution).to(device)

        with torch.no_grad():
            feats = extractor(image)

        base_name = os.path.splitext(img_file)[0]
        for modality, feat in feats.items():
            save_path = os.path.join(output_dir, modality, f'{base_name}.pt')
            torch.save(feat.detach().cpu(), save_path)

    print(f"完成。特征已保存到 {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多模态特征提取")
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='场景目录路径 (需含 images/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出根目录')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_clip', action='store_true', help='禁用 CLIP')
    parser.add_argument('--no_dino', action='store_true', help='禁用 DINOv2')
    parser.add_argument('--use_depth', action='store_true', help='启用 Depth')
    parser.add_argument('--use_normal', action='store_true', help='启用 Normal')
    parser.add_argument('--use_texture', action='store_true', help='启用 Texture')
    parser.add_argument('--resolution', type=int, default=-1,
                        help='最大宽度 (-1 = 自动对 >1080p 降分辨率)')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14',
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'])
    args = parser.parse_args()

    preprocess_scene(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        device=args.device,
        use_clip=not args.no_clip,
        use_dino=not args.no_dino,
        use_depth=args.use_depth,
        use_normal=args.use_normal,
        use_texture=args.use_texture,
        resolution=args.resolution,
        dino_model_name=args.dino_model,
    )
