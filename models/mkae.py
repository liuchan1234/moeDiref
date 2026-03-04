"""
MKAE: Multi-expert Knowledge Augmented Encoder.
论文 III-B Eq.(1)–(4)：多尺度特征金字塔、动态专家路由、内容-风格解耦。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .attention import ContentAwareAttentionModule, AdaptiveAttentionModule
from .experts import MixtureOfExperts


class MultiScalePyramid(nn.Module):
    """
    多尺度特征金字塔 P = {φ_l(I), φ_m(I), φ_h(I)}，论文 Eq.1.
    φ_l: 1×1, φ_m: 3×3, φ_h: 5×5，各输出 base_channels，通道维拼接得 3*base_channels。
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.phi_l = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.phi_m = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.phi_h = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.base_channels = base_channels
        self.out_channels = base_channels * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pl = self.phi_l(x)
        pm = self.phi_m(x)
        ph = self.phi_h(x)
        return torch.cat([pl, pm, ph], dim=1)


class MKAE(nn.Module):
    """
    Multi-expert Knowledge Augmented Encoder.
    流程：I -> 多尺度金字塔 P -> 门控 + N 专家 -> f_fusion -> CnAM -> z_c, AdaAttn -> z_s.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 64,
        num_experts: int = 4,
        expert_blocks: int = 2,
    ):
        super().__init__()
        self.pyramid = MultiScalePyramid(in_channels=in_channels, base_channels=base_channels)
        pyramid_channels = self.pyramid.out_channels
        self.moe = MixtureOfExperts(
            pyramid_channels=pyramid_channels,
            out_channels=out_channels,
            num_experts=num_experts,
            expert_blocks=expert_blocks,
        )
        self.cnam = ContentAwareAttentionModule(channels=out_channels)
        self.ada_attn = AdaptiveAttentionModule(channels=out_channels)
        self.out_channels = out_channels

    def forward(
        self,
        x: torch.Tensor,
        gate_input: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, in_channels, H, W]
        returns: z_c (content), z_s (style)，均为 [B, out_channels, H, W]
        """
        P = self.pyramid(x)
        f_fusion = self.moe(P, gate_input=gate_input, noise_std=noise_std)
        z_c = self.cnam(f_fusion)
        z_s = self.ada_attn(f_fusion)
        return z_c, z_s
