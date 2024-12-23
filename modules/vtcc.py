import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

import matplotlib.pyplot as plt
from .vision_transformer import VisionTransformer
from timm.layers.helpers import to_2tuple
from timm.layers import PatchEmbed



class ViT_VTCC(VisionTransformer):
    """ 
    Vision Transformer Backbone, from MoCov3, Chen et al. https://arxiv.org/abs/2104.02057
    """
    def __init__(self, stop_grad_conv1=False,  **kwargs):
        super().__init__(**kwargs)

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()
        self.dim = self.embed_dim
        
        # weight initialization
        for name, m in self.named_modules():#QKV权重初始化
            if isinstance(m, nn.Linear):#检查全连接层
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))#Kaiming初始化，适用于relu函数
                    nn.init.uniform_(m.weight, -val, val)#以均匀分布的方式初始化线性层的权重
                else:
                    nn.init.xavier_uniform_(m.weight)#Xavier均匀初始化，在保持输入和输出方差一致方面表现良好，有助于保持激活函数的输出在一个合理的范围内。
                nn.init.zeros_(m.bias)#将线性层的偏置初始化为0
        nn.init.normal_(self.cls_token, std=1e-6)#这种接近零的初始化可以使其在训练初期不对结果产生太大影响

    def build_2d_sincos_position_embedding(self, temperature=10000.):#2D正弦余弦位置嵌入
        h, w = self.patch_embed.grid_size#补丁网格的高度和宽度
        grid_w = torch.arange(w, dtype=torch.float32)#196
        grid_h = torch.arange(h, dtype=torch.float32)#196
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'#两个正弦和两个余弦
        pos_dim = self.embed_dim // 4#计算每个方向上的位置编码维度,96
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim#均匀分布的len(pos_dim)张量
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])#（196，96）    
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])#（196，96）
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]#位置编码[1, 196, 384]
        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False #固定位置嵌入


class ConvStem(nn.Module):#卷积茎模块
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8#emded_dim=384
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)#卷积后得到[128,384,14,14]，输入通道3，第一个输出48，第一变为[128,48,112,112]以此类推
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC，展平后得到[128,196,384]      
        x = self.norm(x)#[128,196,384]未出现变化
        return x


def vit_tiny(**kwargs):
    model = ViT_VTCC(
        patch_size=16, embed_dim=192, depth=4, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    return model

def vit_small(**kwargs):
    model = ViT_VTCC(
        patch_size=16, embed_dim=512, depth=8, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    return model

def vit_base(**kwargs):
    model = ViT_VTCC(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    return model
