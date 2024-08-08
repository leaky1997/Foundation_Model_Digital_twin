# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import math
import logging
from torch.nn.modules.container import Sequential

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import collections.abc
from functools import partial


ACTIVATION = {'gelu':nn.GELU(),
              'tanh':nn.Tanh(),
              'sigmoid':nn.Sigmoid(),
              'relu':nn.ReLU(),
              'leaky_relu':nn.LeakyReLU(0.1),
              'softplus':nn.Softplus(),
              'ELU':nn.ELU(),
              'silu':nn.SiLU()}

#%% 1. embedding
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset:offset+x.size(2)]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"  # TODO ?
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1] # b,c,l
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#%% 2. MLP block
class DynamicLinear(nn.Module):
    
    """
    A dynamic linear layer that can interpolate the weight size to support any given input and output feature dimension.
    """

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super(DynamicLinear, self).__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required !!!"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features):
        """
        Forward pass for the dynamic linear layer.
        """
        fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in:]
        this_bias = self.bias
        in_features = x.shape[-1]

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features-self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
                    out_features, self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)
def to_2tuple(x):
    from itertools import repeat
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        # if mlp_layer is DynamicLinearMlp:
        #     self.mlp = mlp_layer(
        #         in_features=dim,
        #         hidden_features=int(dim * mlp_ratio),
        #         act_layer=act_layer,
        #         drop=proj_drop,
        #         prefix_token_length=prefix_token_length,
        #     )
        # else:
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#%% head

class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=d_mid, mlp_ratio=8, mlp_layer=Mlp,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                            prefix_token_length=None)

    def forward(self, x, category_token=None, return_feature=False):
        x = self.proj_in(x)
        B, V, L, C = x.shape
        x = x.view(-1, L, C)
        cls_token = x[:, -1:]
        cls_token = self.cross_att(x, query=cls_token)
        cls_token = cls_token.reshape(B, V, -1, C)

        cls_token = self.mlp(cls_token)
        if return_feature:
            return cls_token
        m = category_token.shape[2]
        cls_token = cls_token.expand(B, V, m, C)
        distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token)

        distance = distance.mean(dim=1)
        return distance


class ForecastHead(nn.Module): # TODO
    def __init__(self, d_model, patch_len, stride, pad, head_dropout=0, prefix_token_length=None):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=int(d_model * 4),
            act_layer=nn.GELU,
            drop=head_dropout,
        )
        self.proj_out = nn.Linear(d_model, patch_len)
        self.pad = pad
        self.patch_len = patch_len
        self.stride = stride
        self.pos_proj = DynamicLinear(
            in_features=128, out_features=128, fixed_in=prefix_token_length)

    def forward(self, x_full, pred_len, token_len):
        x_full = self.proj_in(x_full)
        x_pred = x_full[:, :, -token_len:]
        x = x_full.transpose(-1, -2)
        x = self.pos_proj(x, token_len)
        x = x.transpose(-1, -2)
        x = x + x_pred
        x = self.mlp(x)
        x = self.proj_out(x)

        bs, n_vars = x.shape[0], x.shape[1]
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.fold(x, output_size=(
            pred_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        x = x.squeeze(dim=-1)
        x = x.reshape(bs, n_vars, -1)
        x = x.permute(0, 2, 1)
        return x

#%% backbone

class AFNO1D(nn.Module):
    """
    AFNO2D: 2D Adaptive Fourier Neural Operator
    Args:
        width (int): Channel dimension size.
        num_blocks (int): Number of blocks in block diagonal weight matrices.
        channel_first (bool): If True, input tensor is in (N, C, H, W) format.
        sparsity_threshold (float): Threshold for sparsity.
        modes (int): Number of Fourier modes to keep.
        hidden_size_factor (int): Factor to scale hidden size.
        act (str): Activation function to use.
    """
    def __init__(self, width=512,
                 num_blocks=8,
                 channel_first=False,
                 sparsity_threshold=0.01,
                 modes=32,
                 hidden_size_factor=1,
                 hard_thresholding_fraction=1,
                 act='gelu'):
        super().__init__() # width = dim
        assert width % num_blocks == 0, f"hidden_size {width} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)
        self.act = ACTIVATION[act]

        self.hard_thresholding_fraction = hard_thresholding_fraction
        
        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    def forward(self, x):
        if self.channel_first:
            B, C, L, P = x.shape
            x = x.permute(0, 2, 3, 1)  # -> N, L, P, C
        else:
            B, L, P, C = x.shape
        dtype = x.dtype
        
        x_orig = x
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        
        # x = x.permute(0,1,3,2) # -> N, L, C,P 
        x = rearrange(x, 'b l p c -> b l c p') # -> N, L, C*P
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size) # b,l,c,blocl,block_size

        o1_real = torch.zeros(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor, device=x.device)
        o1_imag = torch.zeros(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor, device=x.device)
        o2_real = torch.zeros_like(x)
        o2_imag = torch.zeros_like(x)

        kept_modes = int(x.shape[1] * self.hard_thresholding_fraction)

        o1_real[:, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) +
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1).to(torch.float32)
        
        # x = F.softshrink(x, lambd=self.sparsity_threshold) # TODO dynamic
        
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], P)
        x = torch.fft.irfft(x, dim=1, norm="ortho")

        x = rearrange(x, 'b l c p -> b l p c') 
        
        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)  # -> N, C, H, W

        x = x.type(dtype)
        
        return x


class Block(nn.Module):
    def __init__(self, mixing_type='afno',
                 double_skip=True,
                 width=32,
                 n_blocks=4,
                 mlp_ratio=1.,
                 channel_first=True,
                 modes=32,
                 act='gelu',
                dim  = 512, # d_model
            prefix_token_length=0,
                 ):
        super().__init__()
        
        self.width = dim
        self.modes = modes
        self.act = ACTIVATION[act]
        self.double_skip = double_skip

        # Normalization layers
        # self.norm1 = torch.nn.GroupNorm(num_groups = 8, num_channels = width)
        # self.norm2 = torch.nn.GroupNorm(num_groups = 8, num_channels = width)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Mixing layer
        if mixing_type == "afno":
            self.filter = AFNO1D(
                width=dim, 
                num_blocks=n_blocks, 
                sparsity_threshold=0.01, 
                channel_first=channel_first, 
                modes=modes,
                hidden_size_factor=1, 
                act=act
            )

        # MLP layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=mlp_hidden_dim),
            self.act,
            nn.Linear(in_features=mlp_hidden_dim, out_features=dim),
        )
        # self.dynamic_mlp = DynamicLinear(
        #     in_features=128, out_features=128, fixed_in=prefix_token_length)
        # self.dynamic_mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, mlp_layer=DynamicLinearMlp)
        
        
    def forward(self, x, prefix_seq_len, attn_mask):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        # x = self.dynamic_mlp(x, prefix_seq_len=prefix_seq_len) # B,C,L,D

        return x 


    def extra_repr(self) -> str:

        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr
