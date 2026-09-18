"""STEGO 无监督纹理/分割特征提取器。

输出 STEGO 中间层的纹理特征。

Reference:
    Hamilton et al., "Unsupervised Semantic Segmentation by Distilling
    Feature Correspondences", ICLR 2022.
    https://arxiv.org/abs/2207.05026
    https://github.com/hamarb172/STEGO

注意: 需要 third_party/STEGO 和训练好的 checkpoint。
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


_STEGO_MEAN = [0.485, 0.456, 0.406]
_STEGO_STD = [0.229, 0.224, 0.225]


class TextureExtractor(nn.Module):
    """STEGO 纹理特征提取器。

    Args:
        ckpt_path: STEGO checkpoint 路径
        repo_path: STEGO 仓库路径
    """

    def __init__(self,
                 ckpt_path='third_party/STEGO/logs/redirected_latest.pt',
                 repo_path='third_party/STEGO'):
        super().__init__()
        repo_abs = os.path.abspath(repo_path)
        if not os.path.isdir(repo_abs):
            raise ImportError(
                f"STEGO 仓库未找到: {repo_abs}。"
                "请先 clone 到 third_party/STEGO。"
            )
        sys.path.insert(0, repo_abs)
        from stego import STEGOModel

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"STEGO checkpoint 未找到: {ckpt_path}")
        self.model = STEGOModel.load_from_checkpoint(ckpt_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 64

        import torchvision
        self.normalize = torchvision.transforms.Normalize(
            mean=_STEGO_MEAN, std=_STEGO_STD
        )

    @torch.no_grad()
    def forward(self, image):
        """提取纹理特征。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
        Returns:
            texture: [B, 64, H, W] 纹理特征
        """
        x = self.normalize(image)
        feats = self.model.forward_features(x)  # [B, 64, H//4, W//4]
        feats = F.interpolate(feats, size=image.shape[-2:], mode='bilinear')
        return feats
