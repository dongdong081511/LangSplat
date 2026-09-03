"""深度诊断"""
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 问题1: 检查图片对应关系
print("=== 问题1: 图片对应关系 ===")
print("原始数据集图片:")
orig_imgs = sorted(os.listdir('dataset/3D Open-vocabulary Segmentation datasets/sofa/images/'))
print(f"  {orig_imgs}")

print("\nsofa_data图片:")
sofa_imgs = sorted(os.listdir('output/sofa_data/images/'))
print(f"  共{len(sofa_imgs)}张: {sofa_imgs[:5]}...{sofa_imgs[-3:]}")

# 原始图片名 00.jpg, 02.jpg, 03.jpg, 06.jpg, 16.jpg
# GT frame_00003 对应哪张原始图?
print("\n分析GT frame_00003对应关系:")
print("  GT JSON说是 'frame_00003.jpg'")
print("  原始数据集有: 00, 02, 03, 06, 16")
print("  推测: frame_00003 可能对应原始 03.jpg")
print("  但sofa_data有28张图，索引不同!")

# 问题2: 检查GT mask的polygon
print("\n=== 问题2: GT mask polygon检查 ===")
with open('output/sofa_data/segmentations_json_v3/frame_00003.json') as f:
    data = json.load(f)

for i, obj in enumerate(data.get('objects', [])):
    cat = obj.get('category', '')
    seg = obj.get('segmentation', [])
    bbox = obj.get('bbox', [])
    
    print(f"\n对象{i}: '{cat}'")
    print(f"  polygon点数: {len(seg)}")
    if seg:
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        print(f"  x范围: [{min(xs)}, {max(xs)}]")
        print(f"  y范围: [{min(ys)}, {max(ys)}]")
    if bbox:
        print(f"  bbox: {bbox}")
    
    # 检查是否是整个图片
    h, w = data['info']['height'], data['info']['width']
    if seg:
        # 如果polygon覆盖整个图片
        if min(xs) <= 1 and max(xs) >= w-1 and min(ys) <= 1 and max(ys) >= h-1:
            print(f"  ⚠️ polygon可能覆盖整个图片!")

# 问题3: 检查原始segmentation格式
print("\n=== 问题3: 原始segmentation格式 ===")
import glob
seg_dirs = sorted(glob.glob('output/sofa_data/segmentations/*/'))
print(f"segmentation子目录: {[os.path.basename(os.path.dirname(d)) for d in seg_dirs]}")

# 检查一个PNG mask
png_files = sorted(glob.glob('output/sofa_data/segmentations/02/*.png'))
if png_files:
    print(f"\n检查PNG mask: {png_files[0]}")
    mask = np.array(Image.open(png_files[0]))
    print(f"  形状: {mask.shape}")
    print(f"  唯一值: {np.unique(mask)}")
    print(f"  非零像素: {(mask > 0).sum()}, 占比: {(mask > 0).mean()*100:.2f}%")

# 检查原始数据集segmentation
print("\n=== 原始数据集segmentation ===")
orig_seg_files = sorted(glob.glob('dataset/3D Open-vocabulary Segmentation datasets/sofa/segmentations/02/*.png'))
if orig_seg_files:
    print(f"原始数据集PNG: {orig_seg_files[0]}")
    mask = np.array(Image.open(orig_seg_files[0]))
    print(f"  形状: {mask.shape}")
    print(f"  唯一值: {np.unique(mask)}")

# 检查图片索引映射
print("\n=== 建立图片索引映射 ===")
# sofa_data图片名: 00.jpg, 01.jpg, ... 27.jpg
# 原始数据集图片名: 00.jpg, 02.jpg, 03.jpg, 06.jpg, 16.jpg
# 需要找出: 原始03.jpg在sofa_data中是第几张

orig_names = ['00.jpg', '02.jpg', '03.jpg', '06.jpg', '16.jpg']
for orig_name in orig_names:
    if orig_name in sofa_imgs:
        idx = sofa_imgs.index(orig_name)
        print(f"原始 {orig_name} -> sofa_data索引 {idx} -> 渲染文件 {idx:05d}.npy")
