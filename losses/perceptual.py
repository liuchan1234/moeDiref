"""
感知损失 L_perc：VGG 多层特征 L1，论文 III-E Eq.19。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

try:
    from torchvision.models import vgg16, VGG16_Weights
except Exception:
    vgg16 = None
    VGG16_Weights = None


_LAYER_INDICES = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}


class VGG16PerceptualLoss(nn.Module):
    """VGG16 特征提取 + 多层 L1 损失。"""

    DEFAULT_LAYERS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

    def __init__(
        self,
        layer_names: Optional[List[str]] = None,
        weight_initialized: bool = True,
    ):
        super().__init__()
        self.layer_names = layer_names or self.DEFAULT_LAYERS
        self.indices = sorted([_LAYER_INDICES[n] for n in self.layer_names if n in _LAYER_INDICES])
        if vgg16 is None:
            self.register_buffer("_dummy", torch.zeros(1))
            self.features = None
            return
        weights = VGG16_Weights.IMAGENET1K_V1 if weight_initialized and VGG16_Weights else None
        vgg = vgg16(weights=weights)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.features = vgg.features

    def forward(
        self,
        I_gt: torch.Tensor,
        I_pred: torch.Tensor,
    ) -> torch.Tensor:
        """L_perc = sum_l ||φ_l(I_gt) - φ_l(I_pred)||_1。"""
        if self.features is None:
            return torch.tensor(0.0, device=I_gt.device, dtype=I_gt.dtype)
        feat_gt = []
        feat_pred = []
        h_gt, h_pred = I_gt, I_pred
        for i, layer in enumerate(self.features):
            h_gt = layer(h_gt)
            h_pred = layer(h_pred)
            if i in self.indices:
                feat_gt.append(h_gt)
                feat_pred.append(h_pred)
        loss = torch.tensor(0.0, device=I_gt.device, dtype=I_gt.dtype)
        for a, b in zip(feat_gt, feat_pred):
            loss = loss + F.l1_loss(a, b)
        return loss / max(len(feat_gt), 1)


def perceptual_loss(
    I_gt: torch.Tensor,
    I_pred: torch.Tensor,
    vgg_module: Optional[VGG16PerceptualLoss] = None,
) -> torch.Tensor:
    """若未传入 vgg_module 则临时创建（无预训练权重时退化为 0）。"""
    if vgg_module is not None:
        return vgg_module(I_gt, I_pred)
    mod = VGG16PerceptualLoss(weight_initialized=True)
    mod = mod.to(I_gt.device).eval()
    return mod(I_gt, I_pred)
