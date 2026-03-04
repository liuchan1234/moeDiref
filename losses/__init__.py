"""
MoE-DiReF 损失模块。
"""

from .reconstruction import reconstruction_loss
from .perceptual import VGG16PerceptualLoss, perceptual_loss
from .adversarial import (
    PatchDiscriminator,
    gradient_penalty,
    wgan_gp_d_loss,
    wgan_gp_g_loss,
)

__all__ = [
    "reconstruction_loss",
    "VGG16PerceptualLoss",
    "perceptual_loss",
    "PatchDiscriminator",
    "gradient_penalty",
    "wgan_gp_d_loss",
    "wgan_gp_g_loss",
]
