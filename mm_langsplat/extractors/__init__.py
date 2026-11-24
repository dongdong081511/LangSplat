"""多模态特征提取器包。

提供 CLIP、DINOv2、Depth、Normal、Texture 提取器的统一访问接口。
每个提取器遵循统一接口：
    - 输入: image [B, 3, H, W], 范围 [0, 1]
    - 输出: feature [B, D, H, W] (像素级) 或 per-tile embeddings
"""
import torch
import torch.nn as nn

from .clip_extractor import CLIPExtractor
from .dino_extractor import DINOv2Extractor
from .depth_extractor import DepthExtractor
from .normal_extractor import NormalExtractor
from .texture_extractor import TextureExtractor
from .sam_extractor import SAMExtractor


class MultiModalExtractor(nn.Module):
    """多模态特征提取统一封装。

    聚合 CLIP (语义)、DINOv2 (视觉)、Depth、Normal、Texture 提取器，
    每个模态可独立启用/禁用。

    输出 dict 键: 'clip', 'dino', 'depth', 'normal', 'texture'
    每个值为 [B, D_m, H, W] 像素级特征张量。

    Args:
        use_clip: 启用 CLIP 提取器 (默认 True)
        use_dino: 启用 DINOv2 提取器 (默认 True)
        use_depth: 启用 Depth 提取器 (默认 False, 需要 third_party)
        use_normal: 启用 Normal 提取器 (默认 False, 需要 third_party)
        use_texture: 启用 Texture 提取器 (默认 False, 需要 third_party)
        dino_model_name: DINOv2 模型名 ('dinov2_vitb14' 或 'dinov2_vitl14')
        clip_pretrained: open_clip pretrained 标签
    """

    def __init__(self,
                 use_clip=True,
                 use_dino=True,
                 use_depth=False,
                 use_normal=False,
                 use_texture=False,
                 dino_model_name='dinov2_vitb14',
                 clip_pretrained='laion2b_s34b_b88k'):
        super().__init__()
        self.use_clip = use_clip
        self.use_dino = use_dino
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.use_texture = use_texture

        if use_clip:
            self.clip = CLIPExtractor(pretrained=clip_pretrained)
        if use_dino:
            self.dino = DINOv2Extractor(model_name=dino_model_name)
        if use_depth:
            self.depth = DepthExtractor()
        if use_normal:
            self.normal = NormalExtractor()
        if use_texture:
            self.texture = TextureExtractor()

    @torch.no_grad()
    def forward(self, image, masks=None):
        """提取所有启用模态的特征。

        Args:
            image: [B, 3, H, W], 范围 [0, 1]
            masks: 可选, SAM 的三层级 mask (用于 tile 级聚合)

        Returns:
            dict, 每个启用模态对应一个 [B, D_m, H, W] 张量
        """
        feats = {}
        if self.use_clip:
            feats['clip'] = self.clip(image, masks)
        if self.use_dino:
            feats['dino'] = self.dino.forward_pixel(image)
        if self.use_depth:
            feats['depth'] = self.depth(image)
        if self.use_normal:
            feats['normal'] = self.normal(image)
        if self.use_texture:
            feats['texture'] = self.texture(image)
        return feats

    @property
    def dims(self):
        """返回各模态输出维度字典。"""
        return {
            'clip': 512 if self.use_clip else 0,
            'dino': self.dino.dim_out if self.use_dino else 0,
            'depth': 1 if self.use_depth else 0,
            'normal': 3 if self.use_normal else 0,
            'texture': 64 if self.use_texture else 0,
        }

    @property
    def enabled_modalities(self):
        """已启用模态名列表。"""
        return [k for k, v in self.dims.items() if v > 0]


__all__ = [
    'CLIPExtractor',
    'DINOv2Extractor',
    'DepthExtractor',
    'NormalExtractor',
    'TextureExtractor',
    'SAMExtractor',
    'MultiModalExtractor',
]
