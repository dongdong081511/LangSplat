"""Depth Anything V2 深度提取器。

输出归一化到 [0, 1] 的相对深度图。

Reference:
    Yang et al., "Depth Anything V2", NeurIPS 2024.
    https://arxiv.org/abs/2406.09414
    https://github.com/DepthAnything/Depth-Anything-V2

注意: 需要 third_party/Depth-Anything-V2 和预训练权重。
      若不存在, 实例化时会抛出 ImportError。
"""
import os
import sys
import torch
import torch.nn as nn


class DepthExtractor(nn.Module):
    """Depth Anything V2 相对深度提取器。

    Args:
        model_path: Depth Anything V2 权重路径 (.pth)
        model_type: 编码器大小 ('vits', 'vitb', 'vitl', 'vitg')
        repo_path: Depth-Anything-V2 仓库路径 (加入 sys.path)
    """

    def __init__(self,
                 model_path='pretrained/depth_anything_v2_vitl.pth',
                 model_type='vitl',
                 repo_path='third_party/Depth-Anything-V2'):
        super().__init__()
        repo_abs = os.path.abspath(repo_path)
        if not os.path.isdir(repo_abs):
            raise ImportError(
                f"Depth-Anything-V2 仓库未找到: {repo_abs}。"
                "请先 clone 到 third_party/Depth-Anything-V2。"
            )
        sys.path.insert(0, repo_abs)
        from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(
            encoder=model_type,
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False,
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Depth Anything V2 权重未找到: {model_path}。"
                "请从 https://github.com/DepthAnything/Depth-Anything-V2 下载。"
            )
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 1

    @torch.no_grad()
    def forward(self, image):
        """提取相对深度。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
        Returns:
            depth: [B, 1, H, W] 归一化到 [0, 1] (0=近, 1=远)
        """
        import numpy as np

        B, _, H, W = image.shape
        depths = []
        for b in range(B):
            img = image[b].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img = (img * 255.0).astype('uint8')
            depth = self.model.infer_image(img)  # [H, W] numpy
            depth = torch.from_numpy(depth).float()
            depths.append(depth)
        depth = torch.stack(depths, dim=0).unsqueeze(1)  # [B, 1, H, W]
        # 逐图归一化到 [0, 1]
        depth_flat = depth.view(B, -1)
        d_min = depth_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        d_max = depth_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        return depth.to(image.device)
