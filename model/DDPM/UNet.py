import torch
import torch.nn as nn   
from typing import Union, List, Optional, Tuple, Dict
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class UNet(nn.Module):
    def __init__(self, image_channels: int = 3,
                 n_channels: int = 64, 
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2
                 ) :
        super(UNet, self).__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, (3, 3), padding=(1, 1))

        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []

        out_channels = in_channels = n_channels

        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels, n_channels * 4)

        up = []

        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels 
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels

            if i > 0:
                up.append(Upsample(in_channels))
        
        self.up = nn.MuduleList(up)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, 
                               image_channels, 
                               kernel_size=(3, 3), 
                               padding=(1, 1))
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor) :
        t = self.time_emb(t)
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, h.pop())
            else:
                s = h.pop()
                x = torch.cat((x, s), dim = 1)
                x = m(x, t)
            
        return self.final(self.act(self.norm(x)))
    
class ResidualBlock(nn.Module):
    def __init(self,
               in_channels: int,
               out_channels: int,
               time_channels: int, 
               n_group: int = 32, 
               dropout: float = 0.1):
        
        super(ResidualBlock, self).__init__()

        self.norm1 = nn.GroupNorm(n_group, 
                                  in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))
        

        self.norm2 = nn.GroupNorm(n_group,
                                  out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))


        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=(1, 1))
        else:
            self.skip = nn.Identity()

        self.time_emb = nn.Linear(time_channels,
                                  out_channels)
        
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)


    def forward(self, 
                x: torch.Tensor,
                t: torch.tensor):
        
        h = self.conv1(
            self.act1(
                self.norm1(x)
            )
        )

        h += self.time_emb(
            self.time_act(t)[:, :, None, None]
        )

        h = self.conv2(
            self.dropout(
                self.act2(
                    self.norm2(h)
                )
            )
        )

        return h + self.shortcut(x)
    
class AttentionBlock(nn.Module):
    def __init__(self,
                 n_channels: int,
                 n_heads: int = 1,
                 d_k: int = None,
                 n_groups: int = 32):
        super(AttentionBlock, self).__init__()

        if d_k is None:
            d_k = n_channels
        
        self.norm = nn.GroupNorm(n_groups, 
                                 n_channels)

        self.projection = nn.Linear(n_channels, 
                                    n_heads * d_k * 3)

        self.output = nn.Linear(n_heads * d_k,
                                n_channels)
        
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, 
                x: torch.Tensor,
                t: Union[torch.Tensor, None] = None):
        _ = t
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, 
                                      self.n_heads, 3 * self.d_k)
        q, k, v = qkv.chunk(3, dim=-2)

        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = attn.softmax(dim = 2)

        res = torch.einsum('bijh, bjhd->bihd', attn, v)

        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res += x
        res = res.permute(0, 2, 1).view(batch, n_channels, height, width)

        return res

class DownBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_channels: int, 
                 has_attn: bool):
        
        super(DownBlock, self).__init__()
        self.res = ResidualBlock(in_channels, 
                                 out_channels,
                                 time_channels)
        
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t)
        x = self.attn(x)
        return x
    

class TimeEmbedding(nn.Module):
    def __init__(self, 
                 n_channels: int):
        super(TimeEmbedding, self).__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, 
                              self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels,
                              self.n_channels)
    
    def forward(self, 
                t: torch.Tensor) -> torch.Tensor:
        half_dim = self.n_channels // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class Upsample(nn.Module):
    def __init__(self,
                 n_channels):
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(n_channels,
                                  n_channels,
                                  kernel_size=(4, 4),
                                  stride=(2, 2),
                                  padding=(1, 1))
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor):
        _ = t
        return self.conv(x)
    
class Downsample(nn.Module):
    def __init__(self,
                 n_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(n_channels,
                              n_channels,
                              kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=(1, 1))
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor):
        _ = t
        return self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self,
                 n_channels: int,
                 time_channels: int):
        super(MiddleBlock, self).__init__()
        self.res1 = ResidualBlock(n_channels, 
                                  n_channels,
                                  time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels,
                                  n_channels,
                                  time_channels)
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor):
        
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x