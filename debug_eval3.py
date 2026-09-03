"""确认segmentation映射关系"""
import os
import glob
from PIL import Image
import numpy as np

print("=== 确认segmentation目录与图片的对应关系 ===")

# 检查每个segmentation子目录对应的图片
seg_dirs = ['02', '04', '10', '15', '22']
orig_imgs = sorted(os.listdir('dataset/3D Open-vocabulary Segmentation datasets/sofa/images/'))
print(f"原始图片: {orig_imgs}")

# 方法1: 检查segmentation mask尺寸与图片尺寸是否匹配
print("\n检查segmentation mask尺寸:")
for seg_dir in seg_dirs:
    png_files = glob.glob(f'dataset/3D Open-vocabulary Segmentation datasets/sofa/segmentations/{seg_dir}/*.png')
    if png_files:
        mask = np.array(Image.open(png_files[0]))
        print(f"  目录{seg_dir}: mask尺寸 {mask.shape[:2]}")

print("\n检查原始图片尺寸:")
for img_name in orig_imgs[:3]:
    img = Image.open(f'dataset/3D Open-vocabulary Segmentation datasets/sofa/images/{img_name}')
    print(f"  {img_name}: {img.size}")

# 检查segmentations_json_v3是如何生成的
print("\n=== 分析segmentations_json_v3的来源 ===")
# 查看是否有转换脚本
import glob
convert_scripts = glob.glob('*convert*.py') + glob.glob('*segment*.py')
print(f"可能的转换脚本: {convert_scripts}")

# 检查segmentations_formatted目录
print("\n=== 检查segmentations_formatted ===")
import os
formatted_dir = 'output/sofa_data/segmentations_formatted'
if os.path.exists(formatted_dir):
    for root, dirs, files in os.walk(formatted_dir):
        level = root.replace(formatted_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f'{subindent}{file}')

# 关键问题: 需要建立正确的映射
# 假设: 子目录名可能是原始数据集的某种索引
print("\n=== 尝试建立映射 ===")
# 原始数据集有9张图: 00, 02, 03, 06, 16, 18, 22, 24, 25
# GT JSON有5帧: frame_00003, frame_00005, frame_00011, frame_00016, frame_00023
# segmentations有5个子目录: 02, 04, 10, 15, 22

# 检查GT JSON中的图片名对应原始数据集
import json
json_files = sorted(glob.glob('output/sofa_data/segmentations_json_v3/frame_*.json'))
print(f"\nGT JSON文件与图片对应:")
for jf in json_files:
    with open(jf) as f:
        data = json.load(f)
    name = data.get('info', {}).get('name', '')
    frame_id = int(os.path.basename(jf).replace('frame_', '').replace('.json', ''))
    print(f"  {os.path.basename(jf)} -> 图片名: {name}, frame_id: {frame_id}")

# 检查原始数据集中哪些图有对应的segmentation
print("\n=== 检查哪些原始图片有segmentation ===")
orig_seg_base = 'dataset/3D Open-vocabulary Segmentation datasets/sofa/segmentations'
for seg_dir in sorted(os.listdir(orig_seg_base)):
    if seg_dir.isdigit() or seg_dir in ['classes.txt']:
        continue
    # 检查是否是目录
    seg_path = os.path.join(orig_seg_base, seg_dir)
    if os.path.isdir(seg_path):
        pngs = glob.glob(f'{seg_path}/*.png')
        print(f"  目录 {seg_dir}: {len(pngs)} 个PNG文件")
