"""
图像变换：Resize、归一化、Tensor，与数据增强（训练时可选）。
"""

import torch
import numpy as np
from typing import Tuple, Optional
from torchvision import transforms as T


def get_train_transforms(
    image_size: int,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> T.Compose:
    """训练时：Resize、随机水平翻转、ToTensor、归一化。"""
    t = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
    ]
    if normalize:
        t.append(T.Normalize(mean=mean, std=std))
    return T.Compose(t)


def get_eval_transforms(
    image_size: int,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> T.Compose:
    """评估/推理：Resize、ToTensor、归一化。"""
    t = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]
    if normalize:
        t.append(T.Normalize(mean=mean, std=std))
    return T.Compose(t)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """将 [0,1] 或归一化后的 tensor [C,H,W] 转为 uint8 [H,W,C]。"""
    if x.dim() == 4:
        x = x[0]
    x = x.cpu().numpy()
    if x.min() < 0 or x.max() <= 1.0:
        x = (x * 255).clip(0, 255)
    else:
        x = x.clip(0, 255)
    x = x.transpose(1, 2, 0).astype(np.uint8)
    return x
