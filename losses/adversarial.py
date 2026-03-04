"""
对抗损失 L_adv：WGAN-GP，论文 III-E Eq.18。
G 为 MoE-DiReF，D 为判别器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PatchDiscriminator(nn.Module):
    """PatchGAN 风格判别器：输出空间图，再对图做 mean 得到标量。"""

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        nf = ndf
        for _ in range(n_layers - 1):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers.append(nn.Conv2d(nf_prev, nf, 4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(nf))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(nf, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


def gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP: (||∇D(interp)||_2 - 1)^2。"""
    B = real.shape[0]
    device = real.device
    alpha = torch.rand(B, 1, 1, 1, device=device, dtype=real.dtype)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    d_interp = d_interp.sum()
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(B, -1)
    norm = (grad.norm(2, dim=1) - 1.0) ** 2
    return norm.mean()


def wgan_gp_d_loss(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """D 的损失：-E[D(real)] + E[D(fake)] + lambda_gp * GP。"""
    d_real = D(real)
    d_fake = D(fake.detach())
    if d_real.dim() > 1:
        d_real = d_real.mean()
        d_fake = d_fake.mean()
    loss = -d_real + d_fake + lambda_gp * gradient_penalty(D, real, fake)
    return loss


def wgan_gp_g_loss(D: nn.Module, fake: torch.Tensor) -> torch.Tensor:
    """G 的损失：-E[D(fake)]。"""
    d_fake = D(fake)
    if d_fake.dim() > 1:
        d_fake = d_fake.mean()
    return -d_fake
