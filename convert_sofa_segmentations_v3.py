#!/usr/bin/env python3
"""
将sofa的PNG格式segmentations转换为评估脚本期望的JSON格式 (使用渲染后的分辨率)
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
import shutil

# 渲染后的分辨率 (从log可知被缩放到1080P)
RENDER_H, RENDER_W = 1080, 1440

def mask_to_polygon(mask):
    """将二值mask转换为多边形 - 返回 [[x,y], [x,y], ...] 格式"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.reshape(-1, 2).tolist()
            if len(polygon) >= 3:
                polygons.append(polygon)
    return polygons

def scale_mask(mask, target_h, target_w):
    """缩放mask到目标分辨率"""
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

def create_json_annotation(img_path, masks_dir, output_dir, frame_idx):
    """创建JSON格式的标注文件"""
    # 读取原始图像获取原始尺寸
    img = cv2.imread(str(img_path))
    orig_h, orig_w = img.shape[:2]
    
    # 计算缩放比例
    scale_h = RENDER_H / orig_h
    scale_w = RENDER_W / orig_w
    
    # 获取对应的mask目录
    mask_subdir = masks_dir / f"{frame_idx:02d}"
    
    if not mask_subdir.exists():
        print(f"Warning: No masks for frame {frame_idx}")
        return None
    
    # 读取所有mask文件
    objects = []
    for mask_file in mask_subdir.glob("*.png"):
        label = mask_file.stem
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        # 缩放mask到渲染分辨率
        scaled_mask = scale_mask(mask, RENDER_H, RENDER_W)
        
        # 二值化
        _, binary_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 获取边界框 (在缩放后的坐标系中)
        coords = np.where(binary_mask > 0)
        if len(coords[0]) == 0:
            continue
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 转换为多边形
        polygons = mask_to_polygon(binary_mask)
        
        for polygon in polygons:
            objects.append({
                "category": label,
                "bbox": [[int(x_min), int(y_min)], [int(x_max), int(y_max)]],
                "segmentation": polygon
            })
    
    # 创建JSON结构 - 使用渲染后的分辨率
    json_data = {
        "info": {
            "name": f"frame_{frame_idx+1:05d}.jpg",
            "height": RENDER_H,
            "width": RENDER_W
        },
        "objects": objects
    }
    
    return json_data

def main():
    sofa_data_dir = Path("output/sofa_data")
    images_dir = sofa_data_dir / "images"
    masks_dir = sofa_data_dir / "segmentations"
    
    output_dir = Path("output/sofa_data/segmentations_json_v3")
    output_dir.mkdir(exist_ok=True)
    
    mask_subdirs = sorted([d for d in masks_dir.iterdir() if d.is_dir()])
    print(f"Found {len(mask_subdirs)} frames with masks: {[d.name for d in mask_subdirs]}")
    print(f"Using rendered resolution: {RENDER_H}x{RENDER_W}")
    
    for mask_subdir in mask_subdirs:
        frame_idx = int(mask_subdir.name)
        img_path = images_dir / f"{frame_idx:02d}.jpg"
        
        if not img_path.exists():
            print(f"Warning: Image not found for frame {frame_idx}")
            continue
        
        json_data = create_json_annotation(img_path, masks_dir, output_dir, frame_idx)
        
        if json_data:
            json_path = output_dir / f"frame_{frame_idx+1:05d}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Created {json_path} with {len(json_data['objects'])} objects")
            
            # 复制对应的图像 (使用渲染后的分辨率)
            img = cv2.imread(str(img_path))
            scaled_img = cv2.resize(img, (RENDER_W, RENDER_H))
            dst_img_path = output_dir / f"frame_{frame_idx+1:05d}.jpg"
            cv2.imwrite(str(dst_img_path), scaled_img)
    
    print(f"\nDone! JSON files saved to {output_dir}")

if __name__ == "__main__":
    main()
