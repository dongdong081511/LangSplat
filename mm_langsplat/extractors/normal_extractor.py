"""DSINE 法向量提取器。

输出相机坐标系下的单位法向量。

Reference:
    Bae et al., "DSINE: Revisiting Depth-aware Normal Estimation", ECCV 2024.
    https://arxiv.org/abs/2403.18205
    https://github.com/baegwangbin/DSINE

注意: 需要 third_party/DSINE 和预训练权重。
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


_DSINE_MEAN = [0.485, 0.456, 0.406]
_DSINE_STD = [0.229, 0.224, 0.225]


class NormalExtractor(nn.Module):
    """DSINE 法向量提取器。

    Args:
        model_path: DSINE 权重路径 (.pt)
        repo_path: DSINE 仓库路径
    """

    def __init__(self,
                 model_path='pretrained/dsine.pt',
                 repo_path='third_party/DSINE'):
        super().__init__()
        repo_abs = os.path.abspath(repo_path)
        if not os.path.isdir(repo_abs):
            raise ImportError(
                f"DSINE 仓库未找到: {repo_abs}。"
                "请先 clone 到 third_party/DSINE。"
            )
        sys.path.insert(0, repo_abs)
        from dsine.models import DSINEModel

        self.model = DSINEModel()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"DSINE 权重未找到: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim_out = 3

        import torchvision
        self.normalize = torchvision.transforms.Normalize(
            mean=_DSINE_MEAN, std=_DSINE_STD
        )

    @torch.no_grad()
    def forward(self, image, intrinsics=None):
        """提取法向量。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
            intrinsics: 可选, [B, 3, 3] 相机内参
        Returns:
            normal: [B, 3, H, W] 单位法向量 (相机坐标系)
        """
        B, _, H, W = image.shape
        if intrinsics is None:
            intrinsics = torch.tensor([
                [W / 2, 0, W / 2],
                [0, H / 2, H / 2],
                [0, 0, 1],
            ], dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1).to(image.device)

        x = self.normalize(image)
        normal = self.model(x, intrinsics=intrinsics)  # [B, 3, H, W]
        normal = F.normalize(normal, dim=1)
        return normal
