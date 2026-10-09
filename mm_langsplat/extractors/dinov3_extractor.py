"""DINOv3 特征提取器（ViT-B/16, 768d）。

权重来源: facebook/dinov3-vitb16-pretrain-lvd1689m (HF safetensors 格式, ModelScope 镜像)
加载方式: 官方 facebookresearch/dinov3 repo torch 实现构造模型 + HF→官方键名映射加载
  (transformers 4.55 要求 torch>=2.1, 本环境 torch 2.0.1 不可用, 故手动映射)

键名映射规则 (HF -> 官方):
  embeddings.cls_token            -> cls_token                       (1,1,768)
  embeddings.mask_token           -> mask_token                      (1,1,768) -> (1,768)
  embeddings.register_tokens      -> storage_tokens                  (1,4,768)
  embeddings.patch_embeddings.*   -> patch_embed.proj.*              (Conv2d)
  layer.N.attention.q_proj.*      -> blocks.N.attn.qkv.weight/bias   (cat q,k,v; k_bias=0)
  layer.N.attention.k_proj.weight -> blocks.N.attn.qkv.weight (中段)
  layer.N.attention.v_proj.*      -> blocks.N.attn.qkv.weight/bias (尾段)
  layer.N.attention.o_proj.*      -> blocks.N.attn.proj.*
  layer.N.layer_scale1.lambda1    -> blocks.N.ls1.lambda1
  layer.N.mlp.up_proj.*           -> blocks.N.mlp.fc1.*
  layer.N.mlp.down_proj.*         -> blocks.N.mlp.fc2.*
  layer.N.norm1/norm2             -> blocks.N.norm1/norm2
  norm.*                          -> norm.*
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from safetensors import safe_open

_DINOV3_MEAN = [0.485, 0.456, 0.406]
_DINOV3_STD = [0.229, 0.224, 0.225]
_DINOV3_REPO = '/home/xiedexia/.cache/torch/hub/facebookresearch_dinov3_main'
_DINOV3_CKPT = '/home/xiedexia/.cache/dinov3_vitb16_model.safetensors'


def convert_hf_safetensors_to_official(hf_path=_DINOV3_CKPT):
    """HF transformers 格式 safetensors -> 官方 DINOv3 ViT state_dict。"""
    f = safe_open(hf_path, 'pt')
    sd = {}
    n_layers = 12
    for i in range(n_layers):
        # qkv 合并: weight (2304,768) = cat[q(768),k(768),v(768)]; k bias 置零 (LinearKMaskedBias 恒 mask)
        q_w = f.get_tensor(f'layer.{i}.attention.q_proj.weight')
        k_w = f.get_tensor(f'layer.{i}.attention.k_proj.weight')
        v_w = f.get_tensor(f'layer.{i}.attention.v_proj.weight')
        q_b = f.get_tensor(f'layer.{i}.attention.q_proj.bias')
        v_b = f.get_tensor(f'layer.{i}.attention.v_proj.bias')
        sd[f'blocks.{i}.attn.qkv.weight'] = torch.cat([q_w, k_w, v_w], dim=0)
        sd[f'blocks.{i}.attn.qkv.bias'] = torch.cat([q_b, torch.zeros_like(q_b), v_b], dim=0)
        sd[f'blocks.{i}.attn.proj.weight'] = f.get_tensor(f'layer.{i}.attention.o_proj.weight')
        sd[f'blocks.{i}.attn.proj.bias'] = f.get_tensor(f'layer.{i}.attention.o_proj.bias')
        sd[f'blocks.{i}.ls1.lambda1'] = f.get_tensor(f'layer.{i}.layer_scale1.lambda1')
        sd[f'blocks.{i}.ls2.lambda1'] = f.get_tensor(f'layer.{i}.layer_scale2.lambda1')
        sd[f'blocks.{i}.mlp.fc1.weight'] = f.get_tensor(f'layer.{i}.mlp.up_proj.weight')
        sd[f'blocks.{i}.mlp.fc1.bias'] = f.get_tensor(f'layer.{i}.mlp.up_proj.bias')
        sd[f'blocks.{i}.mlp.fc2.weight'] = f.get_tensor(f'layer.{i}.mlp.down_proj.weight')
        sd[f'blocks.{i}.mlp.fc2.bias'] = f.get_tensor(f'layer.{i}.mlp.down_proj.bias')
        sd[f'blocks.{i}.norm1.weight'] = f.get_tensor(f'layer.{i}.norm1.weight')
        sd[f'blocks.{i}.norm1.bias'] = f.get_tensor(f'layer.{i}.norm1.bias')
        sd[f'blocks.{i}.norm2.weight'] = f.get_tensor(f'layer.{i}.norm2.weight')
        sd[f'blocks.{i}.norm2.bias'] = f.get_tensor(f'layer.{i}.norm2.bias')
    sd['cls_token'] = f.get_tensor('embeddings.cls_token')
    sd['mask_token'] = f.get_tensor('embeddings.mask_token').reshape(1, -1)
    sd['storage_tokens'] = f.get_tensor('embeddings.register_tokens')
    sd['patch_embed.proj.weight'] = f.get_tensor('embeddings.patch_embeddings.weight')
    sd['patch_embed.proj.bias'] = f.get_tensor('embeddings.patch_embeddings.bias')
    sd['norm.weight'] = f.get_tensor('norm.weight')
    sd['norm.bias'] = f.get_tensor('norm.bias')
    return sd


class DINOv3Extractor(nn.Module):
    """DINOv3 ViT-B/16, 输出 tile 级 768 维特征。接口与 DINOv2Extractor 一致。

    Args:
        use_cls_token: True 用 CLS token, False 用 patch tokens 均值 (与 DINOv2 默认一致)
    """

    def __init__(self, use_cls_token=False, verbose=True):
        super().__init__()
        self.model = torch.hub.load(_DINOV3_REPO, 'dinov3_vitb16',
                                    pretrained=False, source='local')
        # LinearKMaskedBias 的 bias_mask: checkpoint 不含此 buffer, 官方值 = q:1, k:0, v:1
        # rope_embed.periods 在构造时由 _init_weights() 用 base=100 算好, 无需覆盖
        for blk in self.model.blocks:
            o = blk.attn.qkv.out_features
            d = o // 3
            blk.attn.qkv.bias_mask = torch.cat([
                torch.ones(d), torch.zeros(d), torch.ones(d)]).to(blk.attn.qkv.bias.device)
        sd = convert_hf_safetensors_to_official()
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # 预期缺键: LayerScale(若 Identity) / rope periods(构造时算好) / bias_mask(手动填)
        def expected_missing(k):
            return ('ls1' in k or 'ls2' in k or 'rope_embed' in k or 'bias_mask' in k)
        missing = [k for k in missing if not expected_missing(k)]
        unexpected = [k for k in unexpected if not expected_missing(k)]
        if missing or unexpected:
            raise RuntimeError(f'DINOv3 权重映射失败: missing={missing[:5]} unexpected={unexpected[:5]}')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch_size = 16
        self.dim_out = self.model.embed_dim
        self.use_cls_token = use_cls_token
        self.normalize = torchvision.transforms.Normalize(
            mean=_DINOV3_MEAN, std=_DINOV3_STD
        )
        if verbose:
            print(f'DINOv3 ViT-B/16 loaded from HF safetensors (dim={self.dim_out})')

    @torch.no_grad()
    def forward(self, tiles):
        """tiles: [N, 3, 224, 224], 范围 [0,1] -> [N, dim_out] 单位化特征。"""
        x = self.normalize(tiles)
        x = x.to(next(self.model.parameters()).device)
        out = self.model.forward_features(x)
        # DINOv3 forward_features 返回 List[Dict] (每网格一项); 单 tile 输入取第一项
        feat_dict = out[0] if isinstance(out, (list, tuple)) else out
        if self.use_cls_token:
            feats = feat_dict['x_norm_clstoken']
        else:
            feats = feat_dict['x_norm_patchtokens'].mean(dim=1)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def forward_batch(self, tiles, batch_size=64):
        results = []
        for i in range(0, len(tiles), batch_size):
            results.append(self.forward(tiles[i:i + batch_size]).cpu())
        return torch.cat(results, dim=0)
