"""
随机 mask 生成（论文 IV-A：20%–80% 缺失率）
Mask 中 1 表示损坏区域，0 表示保留区域。
"""

import torch
import numpy as np
from typing import Tuple, Optional, Literal


def _ensure_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.asarray(x)


def _ensure_tensor(x, device=None) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x
    if device is not None:
        t = t.to(device)
    return t


def random_rectangle_mask(
    height: int,
    width: int,
    mask_ratio: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    随机矩形 mask：在图像内随机放置一个矩形，面积约为 mask_ratio。
    """
    rng = rng or np.random.default_rng()
    mask = np.zeros((height, width), dtype=np.float32)
    area = height * width
    target_area = area * mask_ratio
    # 矩形边长
    w = max(1, int(np.sqrt(target_area * (rng.random() * 0.5 + 0.5))))
    h = max(1, int(target_area / w))
    w = min(w, width)
    h = min(h, height)
    x = rng.integers(0, max(1, width - w + 1))
    y = rng.integers(0, max(1, height - h + 1))
    mask[y : y + h, x : x + w] = 1.0
    return mask


def random_irregular_mask(
    height: int,
    width: int,
    mask_ratio: float,
    rng: Optional[np.random.Generator] = None,
    num_vertices: int = 8,
    brush_width: int = 20,
) -> np.ndarray:
    """
    不规则形状 mask：用随机多边形 + 膨胀近似自然破损。
    """
    rng = rng or np.random.default_rng()
    mask = np.zeros((height, width), dtype=np.float32)
    area = height * width
    target_area = area * mask_ratio
    # 用多个随机矩形/椭圆叠加得到不规则区域
    num_strokes = max(4, int(target_area / (brush_width * brush_width * 2)))
    for _ in range(num_strokes):
        cx = rng.integers(brush_width, max(brush_width + 1, width - brush_width))
        cy = rng.integers(brush_width, max(brush_width + 1, height - brush_width))
        rw = rng.integers(brush_width, brush_width * 3)
        rh = rng.integers(brush_width, brush_width * 3)
        x0 = max(0, cx - rw)
        x1 = min(width, cx + rw)
        y0 = max(0, cy - rh)
        y1 = min(height, cy + rh)
        mask[y0:y1, x0:x1] = 1.0
    # 若面积不足，再随机挖几块
    current_ratio = mask.sum() / area
    while current_ratio < mask_ratio * 0.95 and num_strokes < 50:
        x = rng.integers(0, max(1, width - brush_width))
        y = rng.integers(0, max(1, height - brush_width))
        mask[y : y + brush_width, x : x + brush_width] = 1.0
        current_ratio = mask.sum() / area
        num_strokes += 1
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def random_mask(
    height: int,
    width: int,
    mask_ratio_range: Tuple[float, float] = (0.2, 0.8),
    mask_type: Literal["random", "rectangle", "irregular"] = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    生成一张随机 mask。
    - height, width: 图像高宽
    - mask_ratio_range: (min_ratio, max_ratio)，论文中 20%–80%
    - mask_type: "random" 时在 rectangle 与 irregular 间随机选
    """
    rng = rng or np.random.default_rng()
    low, high = mask_ratio_range
    mask_ratio = float(rng.uniform(low, high))
    if mask_type == "rectangle":
        return random_rectangle_mask(height, width, mask_ratio, rng)
    if mask_type == "irregular":
        return random_irregular_mask(height, width, mask_ratio, rng)
    if mask_type == "random":
        choice = rng.choice(["rectangle", "irregular"])
        if choice == "rectangle":
            return random_rectangle_mask(height, width, mask_ratio, rng)
        return random_irregular_mask(height, width, mask_ratio, rng)
    raise ValueError(f"Unknown mask_type: {mask_type}")


def make_fused_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    noise_std: float = 0.0,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    论文 Eq.12: I_fus = I_in ⊙ (1−M) + M⊙ε
    - image: [B, C, H, W], 原始或已归一化图像
    - mask: [B, 1, H, W] 或 [B, H, W], 1 表示损坏
    - noise_std: ε 的标准差，0 表示用 0 填充
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(image.dtype).to(image.device)
    if mask.shape[2:] != image.shape[2:]:
        mask = torch.nn.functional.interpolate(
            mask, size=image.shape[2:], mode="nearest"
        )
    one_minus_m = 1.0 - mask
    if noise_std <= 0:
        eps = torch.zeros_like(image, device=image.device, dtype=image.dtype)
    else:
        eps = torch.randn_like(image, generator=rng, device=image.device, dtype=image.dtype) * noise_std
    return image * one_minus_m + mask * eps
