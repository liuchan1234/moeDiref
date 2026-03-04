"""
Rectified Flow：线性 ODE 扩散，论文 III-C Eq.(5)–(6)。
速度场 v_θ(z_t, t)、path consistency loss、ODE 采样（Euler/Heun）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """将标量 t ∈ [0,1] 编码为 dim 维向量（sinusoidal + linear）。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    t = t.unsqueeze(-1)
    args = t * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, t], dim=-1)
    return emb


class VelocityNetwork(nn.Module):
    """
    速度场 v_θ(z_t, t)：输入潜在 z_t 与时间 t，输出与 z_t 同形的速度向量。
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 64,
        time_embed_dim: int = 128,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.GroupNorm(min(8, channels * 2), channels * 2),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = nn.Conv2d(channels * 2, in_channels, 3, padding=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C, H, W], t: [B] 或标量，取值 [0, 1]
        returns: velocity [B, C, H, W]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        t_flat = t.flatten()
        t_emb = timestep_embedding(t_flat, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.conv_in(z)
        h = torch.cat([h, t_emb.expand(-1, -1, h.shape[2], h.shape[3])], dim=1)
        for blk in self.blocks:
            h = h + F.silu(blk(h))
        return self.conv_out(h)


def sample_z_t(z_0: torch.Tensor, z_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """线性插值 z_t = (1-t) z_0 + t z_1，论文 Eq.5 对应路径。"""
    if t.dim() == 0:
        t = t.view(1, 1, 1, 1).expand_as(z_0)
    elif t.dim() == 1:
        t = t.view(-1, 1, 1, 1)
    return (1 - t) * z_0 + t * z_1


def path_consistency_loss(
    v_net: nn.Module,
    z_0: torch.Tensor,
    z_1: torch.Tensor,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L_path = E_t [ ||v_θ(z_t, t) − (z_1 − z_0)||^2 ]，论文 Eq.6.
    z_0: 干净潜在 [B,C,H,W], z_1: 噪声潜在 [B,C,H,W]
    """
    B = z_0.shape[0]
    device = z_0.device
    if t is None:
        t = torch.rand(B, device=device, dtype=z_0.dtype)
    z_t = sample_z_t(z_0, z_1, t)
    target = z_1 - z_0
    pred = v_net(z_t, t)
    return F.mse_loss(pred, target)


def sample_ode_euler(
    v_net: nn.Module,
    z_1: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    从 z_1（t=1，噪声）沿 ODE 积分到 z_0（t=0，数据）。
    dz/dt = v_θ(z, t)，Euler 向后：z_{t-dt} = z_t - v_θ(z_t, t) * dt.
    """
    z = z_1
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype) * (1.0 - i * dt)
        v = v_net(z, t)
        z = z - v * dt
    return z


def sample_ode_heun(
    v_net: nn.Module,
    z_1: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    Heun 二阶 ODE 积分：先 Euler 一步得 z_star，再以 (v(z_t) + v(z_star))/2 更新。
    """
    z = z_1
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_cur = 1.0 - i * dt
        t_next = 1.0 - (i + 1) * dt
        t_tensor = torch.ones(z.shape[0], device=z.device, dtype=z.dtype) * t_cur
        v_cur = v_net(z, t_tensor)
        z_star = z - v_cur * dt
        t_next_t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype) * t_next
        v_next = v_net(z_star, t_next_t)
        z = z - 0.5 * (v_cur + v_next) * dt
    return z


class RectifiedFlow(nn.Module):
    """
    Rectified Flow 模块：封装速度场、训练损失与采样接口。
    """

    def __init__(
        self,
        latent_channels: int,
        channels: int = 64,
        time_embed_dim: int = 128,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.v_net = VelocityNetwork(
            in_channels=latent_channels,
            channels=channels,
            time_embed_dim=time_embed_dim,
            num_blocks=num_blocks,
        )
        self.latent_channels = latent_channels

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """返回 v_θ(z, t)。"""
        return self.v_net(z, t)

    def path_loss(
        self,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return path_consistency_loss(self.v_net, z_0, z_1, t)

    def sample(
        self,
        z_1: torch.Tensor,
        num_steps: int = 10,
        solver: str = "euler",
    ) -> torch.Tensor:
        if solver == "heun":
            return sample_ode_heun(self.v_net, z_1, num_steps)
        return sample_ode_euler(self.v_net, z_1, num_steps)
