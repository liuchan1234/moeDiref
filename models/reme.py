"""
REME: DoFe（下采样特征提取）+ UpFu（上采样融合）+ 三阶段校正。
论文 III-D Eq.7–15。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiAttentionFusion


class DoFe(nn.Module):
    """
    Downsampling Feature Extractor.
    三支路：f_stride = Conv_stride(I_fus), f_max = MaxPool(I_fus), f_avg = AvgPool(I_fus)。
    门控融合：f_DoFe = σ(M)⊙f_stride + (1−σ(M))⊙(f_max + f_avg)，论文 Eq.13–14。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.conv_stride = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.proj_max = nn.Conv2d(in_channels, out_channels, 1)
        self.proj_avg = nn.Conv2d(in_channels, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: I_fus [B, in_channels, H, W]
        mask: M [B, 1, H, W]，1 表示损坏区域
        """
        f_stride = self.conv_stride(x)
        f_max = self.proj_max(self.maxpool(x))
        f_avg = self.proj_avg(self.avgpool(x))
        if mask.shape[2:] != f_stride.shape[2:]:
            m = F.interpolate(mask, size=f_stride.shape[2:], mode="nearest")
        else:
            m = mask
        sigma_m = torch.sigmoid(m)
        f_dofe = sigma_m * f_stride + (1 - sigma_m) * (f_max + f_avg)
        return f_dofe


def _make_grid(h: int, w: int, device: torch.device, dtype: torch.dtype, batch: int = 1) -> torch.Tensor:
    """归一化坐标网格 [1, H, W, 2]，范围 [-1, 1]。"""
    y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
    return grid


class UpFu(nn.Module):
    """
    Upsampling Fusion Unit：空间对齐 + 多注意力融合，论文 Eq.7–11。
    输入 f_high 与 f_denoised_low，输出 f_up_fusion（同通道数），用于 x_constrained_low = x_denoised_low + γ·UpSample(f_up_fusion)。
    """

    def __init__(self, channels_high: int, channels_low: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels_high
        self.channels_high = channels_high
        self.channels_low = channels_low
        self.out_channels = out_channels
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels_high + channels_low, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
        )
        self.proj_high = nn.Conv2d(channels_high, out_channels, 1) if channels_high != out_channels else nn.Identity()
        self.multi_attn = MultiAttentionFusion(out_channels)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        f_high: torch.Tensor,
        f_denoised_low: torch.Tensor,
    ) -> torch.Tensor:
        """
        f_high: [B, C_high, H, W]
        f_denoised_low: [B, C_low, h, w]，与 f_high 可不同分辨率
        输出 f_up_fusion: [B, out_channels, h, w]，与 f_denoised_low 同分辨率
        """
        B, _, H, W = f_high.shape
        _, _, h, w = f_denoised_low.shape
        if (H, W) != (h, w):
            f_high_resized = F.interpolate(f_high, size=(h, w), mode="bilinear", align_corners=False)
        else:
            f_high_resized = f_high
        cat = torch.cat([f_high_resized, f_denoised_low], dim=1)
        offset = self.offset_net(cat)
        grid_base = _make_grid(h, w, f_high.device, f_high.dtype, B)
        scale = torch.tensor([2.0 / max(w, 1), 2.0 / max(h, 1)], device=f_high.device, dtype=f_high.dtype)
        scale = scale.view(1, 1, 1, 2)
        grid = grid_base + offset.permute(0, 2, 3, 1) * scale
        f_aligned = F.grid_sample(f_high, grid, mode="bilinear", padding_mode="border", align_corners=False)
        if f_aligned.shape[2:] != (h, w):
            f_aligned = F.interpolate(f_aligned, size=(h, w), mode="bilinear", align_corners=False)
        f_aligned = self.proj_high(f_aligned) if not isinstance(self.proj_high, nn.Identity) else f_aligned
        f_up_fusion = self.multi_attn(f_aligned)
        return f_up_fusion

    def apply_constraint(
        self,
        x_denoised_low: torch.Tensor,
        f_up_fusion: torch.Tensor,
    ) -> torch.Tensor:
        """x_constrained_low = x_denoised_low + γ·f_up_fusion，论文 Eq.10。"""
        if f_up_fusion.shape[2:] != x_denoised_low.shape[2:]:
            f_up = F.interpolate(f_up_fusion, size=x_denoised_low.shape[2:], mode="bilinear", align_corners=False)
        else:
            f_up = f_up_fusion
        if f_up.shape[1] != x_denoised_low.shape[1]:
            f_up = F.pad(f_up, (0, 0, 0, 0, 0, x_denoised_low.shape[1] - f_up.shape[1]))
        return x_denoised_low + self.gamma * f_up


class REME(nn.Module):
    """
    REME 三阶段校正模块。
    阶段 1：I_fus + M -> DoFe -> f_DoFe
    阶段 2：f_DoFe -> MKAE（外部）-> z_c, z_s
    阶段 3：潜在约束 z_c_t = z_t + λ·(z_c + z_s)，论文 Eq.15
    """

    def __init__(
        self,
        dofe_in_channels: int = 3,
        dofe_out_channels: int = 64,
        upfu_channels_high: Optional[int] = None,
        upfu_channels_low: Optional[int] = None,
        latent_channels: int = 64,
        mkae_out_channels: Optional[int] = None,
    ):
        super().__init__()
        upfu_channels_high = upfu_channels_high or dofe_out_channels
        upfu_channels_low = upfu_channels_low or dofe_out_channels
        mkae_out = mkae_out_channels or dofe_out_channels
        self.dofe = DoFe(in_channels=dofe_in_channels, out_channels=dofe_out_channels)
        self.upfu = UpFu(channels_high=upfu_channels_high, channels_low=upfu_channels_low, out_channels=upfu_channels_high)
        self.latent_channels = latent_channels
        self.dofe_out_channels = dofe_out_channels
        self.proj_content_style = nn.Sequential(
            nn.Conv2d(mkae_out * 2, latent_channels, 1),
            nn.GroupNorm(min(8, latent_channels), latent_channels),
            nn.SiLU(),
        )
        self.lambda_constraint = nn.Parameter(torch.tensor(0.1))

    def forward_dofe(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """阶段 1：I_fus + M -> f_DoFe。"""
        return self.dofe(x, mask)

    def forward_upfu(
        self,
        f_high: torch.Tensor,
        f_denoised_low: torch.Tensor,
    ) -> torch.Tensor:
        """UpFu：f_high 与 f_denoised_low 融合，返回 f_up_fusion。"""
        return self.upfu(f_high, f_denoised_low)

    def constrain_latent(
        self,
        z_t: torch.Tensor,
        z_c: torch.Tensor,
        z_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        阶段 3：z_c_t = z_t + λ·(z_c + z_s)，论文 Eq.15。
        z_c/z_s 来自 MKAE，会投影到 latent 通道并对齐 z_t 的尺寸。
        """
        z_cs = torch.cat([z_c, z_s], dim=1)
        z_cs = self.proj_content_style(z_cs)
        if z_cs.shape[2:] != z_t.shape[2:]:
            z_cs = F.interpolate(z_cs, size=z_t.shape[2:], mode="bilinear", align_corners=False)
        if z_cs.shape[1] != z_t.shape[1]:
            if z_cs.shape[1] < z_t.shape[1]:
                z_cs = F.pad(z_cs, (0, 0, 0, 0, 0, z_t.shape[1] - z_cs.shape[1]))
            else:
                z_cs = z_cs[:, : z_t.shape[1]]
        return z_t + self.lambda_constraint * z_cs

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        z_c: torch.Tensor,
        z_s: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """给定 I_fus(x)、M、外部 MKAE 的 z_c/z_s 与当前 z_t，返回约束后的潜在 z_c_t。"""
        return self.constrain_latent(z_t, z_c, z_s)
