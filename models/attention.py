"""
注意力模块：SAM、CAM、GAT（UpFu 多注意力融合）、CnAM、AdaAttn（MKAE 内容-风格解耦）。
论文 III-B Eq.4, III-D Eq.9-11。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttentionModule(nn.Module):
    """
    CAM (Channel Attention Module).
    对通道维做注意力：全局池化 -> 共享 MLP -> sigmoid -> 通道加权。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = self.avg_pool(x)
        max_out = self.max_pool(x)
        cat = torch.cat([avg, max_out], dim=1)
        w = torch.sigmoid(self.fc(cat))
        return x * w


class SpatialAttentionModule(nn.Module):
    """
    SAM (Spatial Attention Module).
    对空间维做注意力：通道维 avg+max pool -> conv -> sigmoid -> 空间加权。
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg, max_out], dim=1)
        w = torch.sigmoid(self.conv(cat))
        return x * w


class GraphAttentionModule(nn.Module):
    """
    GAT (Graph Attention)，图像上简化为通道间注意力。
    将每个通道视为节点：空间池化得 [B,C]，在 C 维做自注意力，再投影回 [B,C] 并广播到空间。
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.q = nn.Linear(channels, channels * mid)
        self.k = nn.Linear(channels, channels * mid)
        self.v = nn.Linear(channels, channels * mid)
        self.out_proj = nn.Linear(mid, 1)
        self.mid = mid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        z = self.pool(x).flatten(1)
        q = self.q(z).view(B, C, self.mid)
        k = self.k(z).view(B, C, self.mid)
        v = self.v(z).view(B, C, self.mid)
        scale = self.mid ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v)
        out = self.out_proj(out).squeeze(-1)
        out = out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        return x + out


class ContentAwareAttentionModule(nn.Module):
    """
    CnAM (Content-Aware Attention Module).
    提取语义内容：空间自注意力，使输出保持 [B,C,H,W]，用于 MKAE 内容分支。
    """

    def __init__(self, channels: int, head_dim: int = 64, num_heads: Optional[int] = None):
        super().__init__()
        self.channels = channels
        if num_heads is not None:
            self.num_heads = min(num_heads, channels)
        else:
            self.num_heads = max(1, channels // head_dim)
            self.num_heads = min(self.num_heads, channels)
        self.head_dim = channels // self.num_heads
        assert self.num_heads * self.head_dim == channels, "channels 需被 num_heads 整除"
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


class AdaptiveAttentionModule(nn.Module):
    """
    AdaAttn (Adaptive Attention).
    建模风格属性（纹理、颜色等）：全局池化 + MLP 得到风格向量，再对特征做仿射调制。
    输出与输入同形 [B,C,H,W]，用于 MKAE 风格分支。
    """

    def __init__(self, channels: int, style_dim: Optional[int] = None):
        super().__init__()
        style_dim = style_dim or channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, channels * 2),
        )
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        style = self.pool(x)
        style = self.mlp(style)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta


class MultiAttentionFusion(nn.Module):
    """
    UpFu 中的多注意力融合：Conv(SAM(f); CAM(f); GAT(f))，论文 Eq.9-11。
    将 SAM、CAM、GAT 三个分支输出在通道维拼接后做 1x1 卷积融合。
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.sam = SpatialAttentionModule()
        self.cam = ChannelAttentionModule(channels, reduction=reduction)
        self.gat = GraphAttentionModule(channels, reduction=reduction)
        self.fusion = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sam_out = self.sam(x)
        cam_out = self.cam(x)
        gat_out = self.gat(x)
        cat = torch.cat([sam_out, cam_out, gat_out], dim=1)
        return self.fusion(cat)
