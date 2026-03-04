"""
数据集：CelebA、CelebA-HQ、敦煌壁画（文件夹图像）。
返回：image (I_gt), mask；I_fus 在训练时用 mask.make_fused_image 构造（论文 Eq.12）。
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List, Literal

from .mask import random_mask
from .transforms import get_train_transforms, get_eval_transforms


def _collect_image_paths(
    root: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> List[str]:
    paths = []
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return paths
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and name.lower().endswith(extensions):
            paths.append(p)
    return paths


def _celeba_identity_list(root: str, list_name: str) -> Optional[List[str]]:
    """CelebA 的 identity 文件如 list_attr_celeba.txt 等，这里仅做按文件名列表划分用。"""
    list_path = os.path.join(root, list_name)
    if not os.path.isfile(list_path):
        return None
    with open(list_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None
    # 第一行是数量，第二行是属性名，之后每行是 文件名 + 属性
    names = []
    for line in lines[2:]:
        parts = line.strip().split()
        if parts:
            names.append(parts[0])
    return names


class InpaintingDataset(Dataset):
    """
    通用图像修复数据集：从目录加载图像，每次 __getitem__ 随机生成 mask。
    - root: 图像根目录（直接放图片，或 CelebA 的 img_align_celeba 路径）
    - image_size: resize 尺寸
    - mask_range: (min_ratio, max_ratio)
    - mask_type: random | rectangle | irregular
    - train: True 用训练变换（含随机翻转），False 用评估变换
    - image_paths: 若给定则不再扫描 root，用于固定 train/val 划分
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        mask_range: Tuple[float, float] = (0.2, 0.8),
        mask_type: Literal["random", "rectangle", "irregular"] = "random",
        train: bool = True,
        image_paths: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.root = os.path.expanduser(root)
        self.image_size = image_size
        self.mask_range = mask_range
        self.mask_type = mask_type
        self.train = train
        self.rng = np.random.default_rng(seed)

        if image_paths is not None:
            self.paths = [p for p in image_paths if os.path.isfile(p)]
        else:
            self.paths = _collect_image_paths(self.root)

        self.transform = get_train_transforms(image_size) if train else get_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        _, h, w = img.shape

        mask = random_mask(
            h, w,
            mask_ratio_range=self.mask_range,
            mask_type=self.mask_type,
            rng=self.rng,
        )
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": img,
            "mask": mask,
            "path": path,
        }


def get_celeba_paths(root: str, train_split: float = 0.95, seed: int = 42):
    """
    CelebA 根目录下通常有 img_align_celeba，返回 (train_paths, val_paths)。
    """
    img_dir = os.path.join(root, "img_align_celeba")
    if not os.path.isdir(img_dir):
        img_dir = root
    paths = _collect_image_paths(img_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_split)
    return paths[:n_train], paths[n_train:]


class SyntheticInpaintingDataset(Dataset):
    """
    合成数据：随机 tensor，用于无真实数据时验证 dataloader 与 mask、I_fus 流程。
    """

    def __init__(
        self,
        num_samples: int = 32,
        image_size: int = 256,
        mask_range: Tuple[float, float] = (0.2, 0.8),
        mask_type: Literal["random", "rectangle", "irregular"] = "random",
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.mask_range = mask_range
        self.mask_type = mask_type
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(42 + idx)
        img = torch.rand(3, self.image_size, self.image_size)
        h, w = self.image_size, self.image_size
        mask = random_mask(
            h, w,
            mask_ratio_range=self.mask_range,
            mask_type=self.mask_type,
            rng=rng,
        )
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return {
            "image": img,
            "mask": mask,
            "path": f"synthetic_{idx}.png",
        }


def build_dataset(
    name: str,
    root: str,
    image_size: int = 256,
    mask_range: Tuple[float, float] = (0.2, 0.8),
    mask_type: str = "random",
    train: bool = True,
    train_split: float = 0.95,
    seed: int = 42,
) -> InpaintingDataset:
    """
    构建数据集。name in ('celeba', 'celeba_hq', 'dunhuang')。
    CelebA/CelebA-HQ 会尝试按 train_split 划分；敦煌同理。
    """
    root = os.path.expanduser(root)
    train_paths, val_paths = None, None
    if name in ("celeba", "celeba_hq"):
        img_dir = os.path.join(root, "img_align_celeba") if name == "celeba" else root
        if os.path.isdir(img_dir):
            all_paths = _collect_image_paths(img_dir)
            if all_paths:
                rng = np.random.default_rng(seed)
                idx = np.arange(len(all_paths))
                rng.shuffle(idx)
                n_train = int(len(idx) * train_split)
                train_paths = [all_paths[i] for i in idx[:n_train]]
                val_paths = [all_paths[i] for i in idx[n_train:]]
    if train_paths is None and val_paths is None:
        paths = _collect_image_paths(root)
        if not paths:
            paths = _collect_image_paths(os.path.join(root, "images"))
        n = len(paths)
        n_train = int(n * train_split)
        rng = np.random.default_rng(seed)
        order = np.arange(n)
        rng.shuffle(order)
        train_paths = [paths[i] for i in order[:n_train]]
        val_paths = [paths[i] for i in order[n_train:]]

    path_list = train_paths if train else (val_paths or train_paths)
    return InpaintingDataset(
        root=root,
        image_size=image_size,
        mask_range=mask_range,
        mask_type=mask_type,
        train=train,
        image_paths=path_list,
        seed=seed if train else seed + 1,
    )
