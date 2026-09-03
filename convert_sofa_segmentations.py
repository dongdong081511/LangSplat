#!/usr/bin/env python3
"""
将sofa的PNG格式segmentations转换为评估脚本期望的JSON格式
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
import shutil

def mask_to_polygon(mask):
    """将二值mask转换为多边形"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:  # 至少3个点
                polygons.append(polygon)
    return polygons

def create_json_annotation(img_path, masks_dir, output_dir, frame_idx):
    """创建JSON格式的标注文件"""
    # 读取图像获取尺寸
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # 获取对应的mask目录
    mask_subdir = masks_dir / f"{frame_idx:02d}"
    
    if not mask_subdir.exists():
        print(f"Warning: No masks for frame {frame_idx}")
        return None
    
    # 读取所有mask文件
    objects = []
    for mask_file in mask_subdir.glob("*.png"):
        label = mask_file.stem  # 使用文件名作为标签
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        # 二值化
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 获取边界框
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
                "bbox": [[int(x_min), int(y_min)], [int(x_max), int(y_max)]],  # [[x1,y1],[x2,y2]]
                "segmentation": polygon
            })
    
    # 创建JSON结构
    json_data = {
        "info": {
            "name": f"frame_{frame_idx+1:05d}.jpg",
            "height": h,
            "width": w
        },
        "objects": objects
    }
    
    return json_data

def main():
    # 源路径
    sofa_data_dir = Path("output/sofa_data")
    images_dir = sofa_data_dir / "images"
    masks_dir = sofa_data_dir / "segmentations"
    
    # 输出路径
    output_dir = Path("output/sofa_data/segmentations_json")
    output_dir.mkdir(exist_ok=True)
    
    # 读取classes.txt
    classes_file = masks_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"Classes: {classes}")
    
    # 获取有mask的帧
    mask_subdirs = sorted([d for d in masks_dir.iterdir() if d.is_dir()])
    print(f"Found {len(mask_subdirs)} frames with masks: {[d.name for d in mask_subdirs]}")
    
    # 处理每个帧
    for mask_subdir in mask_subdirs:
        frame_idx = int(mask_subdir.name)
        img_path = images_dir / f"{frame_idx:02d}.jpg"
        
        if not img_path.exists():
            print(f"Warning: Image not found for frame {frame_idx}")
            continue
        
        # 创建JSON
        json_data = create_json_annotation(img_path, masks_dir, output_dir, frame_idx)
        
        if json_data:
            # 保存JSON
            json_path = output_dir / f"frame_{frame_idx+1:05d}.json"
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Created {json_path} with {len(json_data['objects'])} objects")
            
            # 复制对应的图像
            dst_img_path = output_dir / f"frame_{frame_idx+1:05d}.jpg"
            shutil.copy(str(img_path), str(dst_img_path))
            print(f"Copied {dst_img_path}")
    
    print(f"\nDone! JSON files saved to {output_dir}")

if __name__ == "__main__":
    main()
