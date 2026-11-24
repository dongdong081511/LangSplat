"""CLIP 特征提取器 (OpenCLIP ViT-B/16)。

提供像素级 CLIP 特征和可选的 SAM mask tile 级聚合。
与 LangSplat preprocess.py 中的 CLIP 逻辑保持一致。

Reference:
    Radford et al., "Learning Transferable Visual Models From Natural
    Language Supervision", ICML 2021.
    https://arxiv.org/abs/2103.00020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip 未安装, 请运行: pip install open_clip_torch")


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class CLIPExtractor(nn.Module):
    """OpenCLIP ViT-B/16 提取器。

    两种模式:
      1. 像素级: 整图编码后上采样到 [B, 512, H, W]
      2. Tile 级 (提供 masks 时): 对每个 SAM tile 提取 CLIP embedding

    Args:
        model_name: open_clip 模型名 (默认 'ViT-B-16')
        pretrained: open_clip pretrained 标签 (默认 'laion2b_s34b_b88k',
                    与 LangSplat preprocess.py 一致)
    """

    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k'):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, precision="fp16"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.dim_out = 512
        self.normalize = torchvision.transforms.Normalize(
            mean=_CLIP_MEAN, std=_CLIP_STD
        )

    @torch.no_grad()
    def forward(self, image, masks=None):
        """提取 CLIP 特征。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
            masks: 可选, SAM 的 tile 字典 (LangSplat 格式)

        Returns:
            masks 为 None 时: [B, 512, H, W] 像素级特征
            masks 提供时: per-tile embeddings dict
        """
        if masks is not None:
            return self._extract_with_masks(image, masks)
        return self._extract_pixel(image)

    def _extract_pixel(self, image):
        """像素级 CLIP 特征提取 (整图编码 + 上采样)。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
        Returns:
            [B, 512, H, W] 归一化特征
        """
        x = self.normalize(image)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.half()
        feat = self.model.encode_image(x)  # [B, 512]
        feat = feat.float()
        feat = F.normalize(feat, dim=-1)
        # 广播到空间维度
        feat = feat.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        feat = F.interpolate(feat, size=image.shape[-2:], mode='bilinear')
        return feat  # [B, 512, H, W]

    def _extract_with_masks(self, image, masks):
        """使用 SAM mask 提取 per-tile CLIP embeddings。

        镜像 LangSplat preprocess.py 中的 _embed_clip_sam_tiles 逻辑。

        Args:
            image: [B, 3, H, W]
            masks: dict, 键为 'default'/'s'/'m'/'l', 值为 [N, 3, 224, 224]
        Returns:
            dict, {mode: [N, 512]} 归一化 embeddings
        """
        embeds = {}
        for mode, tiles in masks.items():
            tiles = tiles.to(next(self.model.parameters()).device)
            with torch.no_grad():
                embed = self.model.encode_image(tiles.half())
            embed = embed.float()
            embed = F.normalize(embed, dim=-1)
            embeds[mode] = embed.detach().cpu().half()
        return embeds
