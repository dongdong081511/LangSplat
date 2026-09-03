#!/usr/bin/env python3
"""
深入分析官方sofa数据与我们自提取数据的特征质量差异
"""
import numpy as np
import os
import cv2

print("="*70)
print("特征质量对比分析")
print("="*70)

def load_features(feat_dir):
    """加载特征文件"""
    f_feats = []
    s_feats = []
    feat_files = sorted([f for f in os.listdir(feat_dir) if f.endswith('_f.npy')])
    
    for f_file in feat_files[:10]:
        frame_name = f_file.replace('_f.npy', '')
        f_path = os.path.join(feat_dir, f_file)
        s_path = os.path.join(feat_dir, frame_name + '_s.npy')
        
        if os.path.exists(f_path) and os.path.exists(s_path):
            f_feats.append(np.load(f_path))
            s_feats.append(np.load(s_path))
    
    return f_feats, s_feats, feat_files[:10]

def analyze_feature_distribution(f_feats, name):
    """分析特征分布"""
    all_feats = np.concatenate(f_feats, axis=0)
    norms = np.linalg.norm(all_feats, axis=1)
    
    print(f"\n{name} 特征分布分析:")
    print(f"  总cluster数: {len(all_feats)}")
    print(f"  特征维度: {all_feats.shape[1]}")
    print(f"  特征值范围: [{all_feats.min():.4f}, {all_feats.max():.4f}]")
    print(f"  特征均值: {all_feats.mean():.6f}")
    print(f"  特征标准差: {all_feats.std():.6f}")
    print(f"  L2范数均值: {norms.mean():.4f}")
    print(f"  L2范数标准差: {norms.std():.4f}")
    
    is_normalized = np.allclose(norms, 1.0, atol=0.1)
    print(f"  是否近似归一化: {is_normalized}")
    
    return all_feats, norms

def analyze_seg_maps(s_feats, name):
    """分析seg map"""
    print(f"\n{name} Seg Map分析:")
    total_clusters = []
    for i, s in enumerate(s_feats[:5]):
        print(f"  Frame {i}: shape={s.shape}")
        if len(s.shape) == 3:
            cluster_map = s[0] if s.shape[0] <= 4 else s[:,:,0]
            unique_clusters = len(np.unique(cluster_map))
            total_clusters.append(unique_clusters)
            print(f"    unique clusters: {unique_clusters}")
    if total_clusters:
        print(f"  平均clusters数: {np.mean(total_clusters):.1f}")

# ========== 分析官方sofa数据 ==========
print("\n" + "="*70)
print("1. 官方 sofa 数据分析")
print("="*70)

sofa_feat_dir = 'output/sofa_data/language_features'
sofa_f, sofa_s, sofa_files = load_features(sofa_feat_dir)
sofa_all_feats, sofa_norms = analyze_feature_distribution(sofa_f, "官方sofa")
analyze_seg_maps(sofa_s, "官方sofa")

# ========== 分析我们的数据 ==========
print("\n" + "="*70)
print("2. 我们的数据分析 (dataset/lerf_ovs)")
print("="*70)

# 分析teatime
teatime_feat_dir = 'dataset/lerf_ovs/teatime/language_features'
if os.path.exists(teatime_feat_dir):
    teatime_f, teatime_s, teatime_files = load_features(teatime_feat_dir)
    teatime_all_feats, teatime_norms = analyze_feature_distribution(teatime_f, "lerf_ovs/teatime")
    analyze_seg_maps(teatime_s, "lerf_ovs/teatime")
else:
    print(f"目录不存在: {teatime_feat_dir}")
    teatime_all_feats = None

# 分析ramen
ramen_feat_dir = 'dataset/lerf_ovs/ramen/language_features'
if os.path.exists(ramen_feat_dir):
    ramen_f, ramen_s, ramen_files = load_features(ramen_feat_dir)
    ramen_all_feats, ramen_norms = analyze_feature_distribution(ramen_f, "lerf_ovs/ramen")
    analyze_seg_maps(ramen_s, "lerf_ovs/ramen")
else:
    ramen_all_feats = None

# ========== 关键对比 ==========
print("\n" + "="*70)
print("3. 关键差异对比")
print("="*70)

print("\n【特征统计对比】")
print(f"{'指标':<25} {'官方sofa':<15} {'我们的teatime':<15} {'我们的ramen':<15}")
print("-"*70)
if teatime_all_feats is not None:
    print(f"{'特征均值':<25} {sofa_all_feats.mean():<15.6f} {teatime_all_feats.mean():<15.6f} {ramen_all_feats.mean() if ramen_all_feats is not None else 'N/A':<15}")
    print(f"{'特征标准差':<25} {sofa_all_feats.std():<15.6f} {teatime_all_feats.std():<15.6f} {ramen_all_feats.std() if ramen_all_feats is not None else 'N/A':<15}")
    print(f"{'L2范数均值':<25} {sofa_norms.mean():<15.4f} {teatime_norms.mean():<15.4f} {ramen_norms.mean() if ramen_all_feats is not None else 'N/A':<15}")

# 图像分辨率对比
print("\n【图像分辨率对比】")
sofa_img = cv2.imread('output/sofa_data/images/00.jpg')
teatime_img = cv2.imread('dataset/lerf_ovs/teatime/images/frame_00001.jpg')
ramen_img = cv2.imread('dataset/lerf_ovs/ramen/images/frame_00001.jpg')
print(f"官方sofa: {sofa_img.shape if sofa_img is not None else 'None'}")
print(f"lerf_ovs/teatime: {teatime_img.shape if teatime_img is not None else 'None'}")
print(f"lerf_ovs/ramen: {ramen_img.shape if ramen_img is not None else 'None'}")

# dim3特征对比
print("\n【降维特征(dim3)对比】")
sofa_dim3_dir = 'output/sofa_data/language_features_dim3'
teatime_dim3_dir = 'dataset/lerf_ovs/teatime/language_features_dim3'
ramen_dim3_dir = 'dataset/lerf_ovs/ramen/language_features_dim3'

for name, dim3_dir in [("官方sofa", sofa_dim3_dir), ("teatime", teatime_dim3_dir), ("ramen", ramen_dim3_dir)]:
    if os.path.exists(dim3_dir):
        dim3_files = [f for f in os.listdir(dim3_dir) if f.endswith('_f.npy')]
        if dim3_files:
            dim3 = np.load(os.path.join(dim3_dir, dim3_files[0]))
            print(f"{name}: shape={dim3.shape}, range=[{dim3.min():.4f}, {dim3.max():.4f}]")

# 检查特征提取脚本
print("\n【特征提取脚本参数】")
preprocess_script = 'preprocess.py'
if os.path.exists(preprocess_script):
    print(f"找到: {preprocess_script}")
    with open(preprocess_script, 'r') as f:
        content = f.read()
    import re
    # 查找关键参数
    for param in ['n_segments', 'compactness', 'sigma']:
        if param in content:
            matches = re.findall(rf'{param}\s*[=:]\s*([\d.]+)', content)
            if matches:
                print(f"  {param}: {matches}")
    # 检查CLIP模型
    for model in ['ViT-B-32', 'ViT-L-14', 'ViT-B-16', 'ViT-H-14']:
        if model in content:
            print(f"  CLIP模型: {model}")
            break

print("\n" + "="*70)
print("分析完成!")
print("="*70)
