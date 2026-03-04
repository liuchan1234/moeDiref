"""
专家网络与门控：论文 III-B Eq.2–3。
G(·) 输出 N 维权重且和为 1；Experti(P) 为第 i 个专家，输入多尺度特征 P，输出同维特征。
"""

import torch
import torch.nn as nn
from typing import Optional


class ExpertBlock(nn.Module):
    """
    单个专家：对多尺度特征 P 做卷积与残差，输出与指定通道数相同。
    每个专家为 ResBlock 风格，输入 pyramid_channels，输出 out_channels。
    """

    def __init__(
        self,
        pyramid_channels: int,
        out_channels: int,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.proj_in = nn.Conv2d(pyramid_channels, out_channels, 1)
        layers = []
        for _ in range(num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                )
            )
        self.blocks = nn.ModuleList(layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for blk in self.blocks:
            x = self.relu(x + blk(x))
        return x


class GatingNetwork(nn.Module):
    """
    门控网络 G(·)：输入 I ⊕ ε（图像 + 可选噪声），输出 N 维权重，和为 1（Softmax）。
    论文 Eq.2: w = G(I_in ⊕ ε; θ_g), Σ w_i = 1.
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        mid_channels: int = 64,
        use_noise: bool = True,
    ):
        super().__init__()
        self.use_noise = use_noise
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, num_experts),
        )
        self.num_experts = num_experts

    def forward(
        self,
        x: torch.Tensor,
        noise_std: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, num_experts] 且每行和为 1
        """
        if self.use_noise and noise_std > 0:
            x = x + torch.randn_like(x, generator=generator, device=x.device, dtype=x.dtype) * noise_std
        z = self.pool(x)
        logits = self.mlp(z)
        return logits.softmax(dim=-1)


class MixtureOfExperts(nn.Module):
    """
    N 个专家 + 门控：f_fusion = Σ_i w_i · Expert_i(P)，论文 Eq.3.
    """

    def __init__(
        self,
        pyramid_channels: int,
        out_channels: int,
        num_experts: int,
        expert_blocks: int = 2,
    ):
        super().__init__()
        self.gating = GatingNetwork(
            in_channels=pyramid_channels,
            num_experts=num_experts,
        )
        self.experts = nn.ModuleList([
            ExpertBlock(pyramid_channels, out_channels, num_blocks=expert_blocks)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(
        self,
        pyramid: torch.Tensor,
        gate_input: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        pyramid: [B, pyramid_channels, H, W] 多尺度融合特征
        gate_input: 若提供则用其做门控，否则用 pyramid
        """
        gate_in = gate_input if gate_input is not None else pyramid
        w = self.gating(gate_in, noise_std=noise_std)
        out = torch.zeros_like(self.experts[0](pyramid))
        for i, expert in enumerate(self.experts):
            out = out + w[:, i : i + 1].unsqueeze(-1).unsqueeze(-1) * expert(pyramid)
        return out
