from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape # B=batch_size;N=num_patches + 1(或2);C=total_embed_dim];

        # qkv(): -> [B, N, 3*C]
        # reshape: -> [B, N, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, B, num_heads, N, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2] # q,k和v都为[B, num_heads, N, embed_dim_per_head]

        attn = (q @ k.transpose(-2, -1)) * self.scale # attn为[B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # x为[B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)   # x为[B, N, C]
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)                                                                # Layer Normalization
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,       # Multi-Head Attention
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()       # Drop Depth
        self.norm2 = norm_layer(dim)                                                                # Layer Normalization
        mlp_hidden_dim = int(dim * mlp_ratio)       # mlp_ratio = MLP隐藏层维度 / 嵌入层维度，所以mlp_hidden_dim是MLP隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)   # MLP层

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
            img_size：输入图像大小
            patch_size：图像片大小
            in_c (int)：图像通道数
            num_classes (int)：图像类别
            embed_dim (int)：嵌入层维度
            depth (int)：transform的encoder块数
            num_heads (int)：multi-head attention的头数
            mlp_ratio (int)：mlp_ratio = MLP隐藏层维度 / 嵌入层维度
            qkv_bias (bool)：可学习参数的偏置向量，
            qk_scale (float)：query和key的缩放因子
            representation_size (Optional[int])：和Representation Layer相关
            distilled (bool)：加入了知识蒸馏（knowledge distillation），是为DeiT models，是对ViT models的改进
            drop_ratio (float)：随机丢失率
            attn_drop_ratio (float): 自注意随机丢失率
            drop_path_ratio (float): 层随机丢失
            embed_layer (nn.Module): 类：用来实现图片分块
            norm_layer: (nn.Module): 类：用来实现Layer Normalization
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes                              # 类别数目
        self.num_features = self.embed_dim = embed_dim              # patch的维度
        self.num_tokens = 2 if distilled else 1                     # cls_token数+dist_token数
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # Layer Normalization类
        act_layer = act_layer or nn.GELU                            # 实现激活层

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)    # 实现Embedded Patches
        num_patches = self.patch_embed.num_patches                  # patch数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))                                 # CLS token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None         # Distilled token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))     # Position Embedding
        self.pos_drop = nn.Dropout(p=drop_ratio)                                                    # Position Embedding dropout

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]     # Drop Path 对层随机dropout
        self.blocks = nn.Sequential(*[                                          # depth个Block类实例
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)                                       # Layer Normalization实例

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # x为[B, 196, 768]

        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # cls_token为[B, 1, 768]

        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # x为[B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1) # x为[B, 198, 768]

        x = self.pos_drop(x + self.pos_embed) # x为[B, 197或198, 768]

        x = self.blocks(x) # x为[B, 197或198, 768]

        x = self.norm(x) # x为[B, 197或198, 768]

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])     # 得到所有样本索引为0的序列，即cls_token对应的输出，结果是[B, 768]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x) # x为[B, 768]
        if self.head_dist is not None: # self.head_dist为None
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x) # x为[B, 5]
        return x


def _init_vit_weights(m):
    # m表示第m层
    if isinstance(m, nn.Linear):        # 线性层初始化策略
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):      # 卷积层初始化策略
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):   # Normalization层初始化策略
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    模型类型：ViT-B
    patch的大小：224 x 224
    在ImageNet-21k上进行的预训练
    num_classes：5个类别
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model