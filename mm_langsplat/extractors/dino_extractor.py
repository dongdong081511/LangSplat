"""DINOv2 特征提取器。

支持 tile 级和像素级两种提取模式：
  - forward(tiles):  tile 级, 输入 [N, 3, 224, 224], 输出 [N, dim_out]
  - forward_pixel(image): 像素级, 输入 [B, 3, H, W], 输出 [B, dim_out, H, W]

Reference:
    Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024.
    https://arxiv.org/abs/2304.07193
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


_DINOV2_MEAN = [0.485, 0.456, 0.406]
_DINOV2_STD = [0.229, 0.224, 0.225]


class DINOv2Extractor(nn.Module):
    """DINOv2 ViT-B/14, 输出 tile 级 768 维或像素级特征。

    Args:
        model_name: 'dinov2_vitb14' (768d) 或 'dinov2_vitl14' (1024d)
        local_path: dinov2 仓库本地路径 (Python 3.9 兼容)
        use_cls_token: True 用 CLS token, False 用 patch tokens 均值
    """

    def __init__(self, model_name='dinov2_vitb14',
                 local_path='/home/xiedexia/.cache/torch/hub/facebookresearch_dinov2_main',
                 use_cls_token=False):
        super().__init__()
        self.model = torch.hub.load(local_path, model_name,
                                    verbose=False, source='local')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch_size = 14
        self.dim_out = self.model.embed_dim
        self.use_cls_token = use_cls_token

        self.normalize = torchvision.transforms.Normalize(
            mean=_DINOV2_MEAN, std=_DINOV2_STD
        )

    @torch.no_grad()
    def forward(self, tiles):
        """提取 tile 级 DINOv2 特征。

        Args:
            tiles: [N, 3, 224, 224], 范围 [0, 1]
        Returns:
            features: [N, dim_out] 归一化到单位长度
        """
        x = self.normalize(tiles)
        x = x.to(next(self.model.parameters()).device)

        if self.use_cls_token:
            out = self.model.forward_features(x)
            feats = out['x_norm_clstoken']  # [N, dim_out]
        else:
            out = self.model.forward_features(x)
            patch_tokens = out['x_norm_patchtokens']  # [N, num_patches, dim_out]
            feats = patch_tokens.mean(dim=1)  # [N, dim_out]

        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def forward_batch(self, tiles, batch_size=64):
        """批量 forward, 避免 tile 数量过多时 OOM。

        Args:
            tiles: [N, 3, 224, 224], 范围 [0, 1]
            batch_size: 每批 tile 数
        Returns:
            features: [N, dim_out]
        """
        results = []
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            feat = self.forward(batch)
            results.append(feat.cpu())
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def forward_pixel(self, image):
        """提取像素级 DINOv2 特征 [B, D, H, W]。

        将图像 pad 到 patch_size (14) 的倍数, 提取 patch tokens,
        重排为 2D 网格, 上采样到原始分辨率。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
        Returns:
            features: [B, dim_out, H, W] 像素级特征
        """
        B, _, H, W = image.shape
        device = next(self.model.parameters()).device
        x = self.normalize(image).to(device)

        # Pad H, W 到 patch_size 的倍数
        H_pad = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_pad = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        x = F.interpolate(x, size=(H_pad, W_pad), mode='bilinear', align_corners=False)

        out = self.model.forward_features(x)
        patch_tokens = out['x_norm_patchtokens']  # [B, N, D]

        h_grid = H_pad // self.patch_size
        w_grid = W_pad // self.patch_size
        feats = patch_tokens.transpose(1, 2).reshape(B, -1, h_grid, w_grid)
        feats = F.interpolate(feats, size=(H, W), mode='bilinear', align_corners=False)
        return feats  # [B, D, H, W]
