"""
重建损失 L_rec = ||I_gt − I_pred||_2^2，论文 III-E Eq.17。
"""

import torch
import torch.nn.functional as F
from typing import Optional


def reconstruction_loss(
    I_gt: torch.Tensor,
    I_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L_rec = ||I_gt - I_pred||_2^2。
    若提供 mask（1 为损坏区），可仅对 mask 区域或仅对有效区域计算；默认全图。
    """
    if mask is None:
        return F.mse_loss(I_pred, I_gt)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(I_gt)
    diff = (I_pred - I_gt) ** 2
    return (diff * mask).sum() / (mask.sum() + 1e-8)
