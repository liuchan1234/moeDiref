"""
图像修复评估指标：PSNR、SSIM、FID、LPIPS（论文 IV-A）。
输入图像约定为 [0,1] 或 [-1,1]；内部按需转换。
"""

import os
import torch
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
except Exception:
    inception_v3 = None
    Inception_V3_Weights = None

try:
    import lpips
except Exception:
    lpips = None


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """将 [-1,1] 转到 [0,1]。"""
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def psnr(I_gt: torch.Tensor, I_pred: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR = 10 * log10(max_val^2 / MSE)。单张或 batch 返回标量。"""
    if I_gt.dim() == 3:
        I_gt = I_gt.unsqueeze(0)
        I_pred = I_pred.unsqueeze(0)
    mse = F.mse_loss(I_pred, I_gt)
    if mse.item() < 1e-10:
        return torch.tensor(100.0, device=I_gt.device, dtype=I_gt.dtype)
    return (10.0 * math.log10(max_val ** 2 / mse.item())) * torch.ones(1, device=I_gt.device, dtype=I_gt.dtype).squeeze()


def psnr_batch(I_gt: torch.Tensor, I_pred: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """对 batch 逐张算 PSNR 再平均。"""
    B = I_gt.shape[0]
    psnrs = []
    for i in range(B):
        p = psnr(I_gt[i], I_pred[i], max_val)
        psnrs.append(p.item() if p.numel() == 1 else p)
    return torch.tensor(sum(psnrs) / B, device=I_gt.device, dtype=I_gt.dtype)


def ssim_impl(
    I_gt: torch.Tensor,
    I_pred: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """单通道或三通道 SSIM（简化版，无边界 padding 优化）。"""
    C = I_gt.shape[1]
    if I_gt.dim() == 3:
        I_gt = I_gt.unsqueeze(0)
        I_pred = I_pred.unsqueeze(0)
    pad = window_size // 2
    mu_gt = F.avg_pool2d(I_gt, window_size, stride=1, padding=pad)
    mu_pred = F.avg_pool2d(I_pred, window_size, stride=1, padding=pad)
    sigma_gt_sq = F.avg_pool2d(I_gt ** 2, window_size, stride=1, padding=pad) - mu_gt ** 2
    sigma_pred_sq = F.avg_pool2d(I_pred ** 2, window_size, stride=1, padding=pad) - mu_pred ** 2
    sigma_gt_pred = F.avg_pool2d(I_gt * I_pred, window_size, stride=1, padding=pad) - mu_gt * mu_pred
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (2 * mu_gt * mu_pred + c1) * (2 * sigma_gt_pred + c2) / (
        (mu_gt ** 2 + mu_pred ** 2 + c1) * (sigma_gt_sq + sigma_pred_sq + c2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


def ssim(I_gt: torch.Tensor, I_pred: torch.Tensor) -> torch.Tensor:
    """SSIM，输入 [B,C,H,W] 或 [C,H,W]，返回标量。"""
    return ssim_impl(I_gt, I_pred, window_size=11, size_average=True)


def ssim_batch(I_gt: torch.Tensor, I_pred: torch.Tensor) -> torch.Tensor:
    """对 batch 逐张 SSIM 再平均。"""
    B = I_gt.shape[0]
    vals = []
    for i in range(B):
        vals.append(ssim(I_gt[i : i + 1], I_pred[i : i + 1]).item())
    return torch.tensor(sum(vals) / B, device=I_gt.device, dtype=I_gt.dtype)


def _inception_features(imgs: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """imgs [B,3,H,W] 在 [0,1]，resize 到 299，取 Inception 最后一层前 2048 维特征。"""
    imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
    imgs = (imgs - 0.5) * 2.0
    out = model(imgs)
    if isinstance(out, tuple):
        out = out[0]
    return out.view(out.shape[0], -1)


def _frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
    """Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    try:
        covmean = torch.linalg.sqrtm(sigma1 @ sigma2)
        if covmean.is_complex():
            covmean = covmean.real
        if torch.isnan(covmean).any() or torch.isinf(covmean).any():
            covmean = torch.zeros_like(sigma1)
    except Exception:
        covmean = torch.zeros_like(sigma1)
    return (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)).item()


class FIDComputer:
    """用 Inception-V3 特征计算 FID（最后一层改为 Identity 取 2048 维特征）。"""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        if inception_v3 is not None:
            try:
                w = Inception_V3_Weights.IMAGENET1K_V1 if Inception_V3_Weights else None
                self.model = inception_v3(weights=w).eval().to(device)
                self.model.fc = torch.nn.Identity()
                for p in self.model.parameters():
                    p.requires_grad = False
            except Exception:
                self.model = None

    def get_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs [B,3,H,W] in [0,1]."""
        if self.model is None:
            return torch.zeros(imgs.shape[0], 2048, device=imgs.device)
        return _inception_features(imgs.to(self.device), self.model)

    @staticmethod
    def compute_from_features(feat_real: torch.Tensor, feat_fake: torch.Tensor) -> float:
        """从两组特征向量 [N, D] 计算 FID。"""
        mu_r = feat_real.mean(dim=0)
        mu_f = feat_fake.mean(dim=0)
        nr, nf = feat_real.shape[0], feat_fake.shape[0]
        sigma_r = ((feat_real - mu_r).T @ (feat_real - mu_r)) / (nr - 1) if nr > 1 else torch.zeros(feat_real.shape[1], feat_real.shape[1], device=feat_real.device)
        sigma_f = ((feat_fake - mu_f).T @ (feat_fake - mu_f)) / (nf - 1) if nf > 1 else torch.zeros(feat_fake.shape[1], feat_fake.shape[1], device=feat_fake.device)
        eps = 1e-6 * torch.eye(sigma_r.shape[0], device=sigma_r.device, dtype=sigma_r.dtype)
        return _frechet_distance(mu_r, sigma_r + eps, mu_f, sigma_f + eps)


class LPIPSComputer:
    """LPIPS 感知距离（可选依赖 lpips）。"""

    def __init__(self, device: torch.device, net: str = "vgg"):
        self.device = device
        self.lpips_fn = None
        if lpips is not None:
            try:
                self.lpips_fn = lpips.LPIPS(net=net).to(device).eval()
            except Exception:
                pass

    def __call__(self, I_gt: torch.Tensor, I_pred: torch.Tensor) -> torch.Tensor:
        """输入 [B,3,H,W]，范围 [-1,1] 或 [0,1]（lpips 常用 [-1,1]）。返回标量。"""
        if self.lpips_fn is None:
            return torch.tensor(0.0, device=I_gt.device)
        if I_gt.dim() == 3:
            I_gt = I_gt.unsqueeze(0)
            I_pred = I_pred.unsqueeze(0)
        I_gt = I_gt.to(self.device)
        I_pred = I_pred.to(self.device)
        if I_gt.max() <= 1.0 and I_gt.min() >= 0:
            I_gt = I_gt * 2.0 - 1.0
            I_pred = I_pred * 2.0 - 1.0
        with torch.no_grad():
            d = self.lpips_fn(I_gt, I_pred)
        return d.mean()


def compute_metrics(
    I_gt: torch.Tensor,
    I_pred: torch.Tensor,
    in_01: bool = True,
    fid_computer: Optional[FIDComputer] = None,
    lpips_computer: Optional[LPIPSComputer] = None,
) -> dict:
    """
    汇总 PSNR、SSIM，可选 FID、LPIPS。
    I_gt, I_pred: [B,3,H,W]，若 in_01=False 则视为 [-1,1] 会先转到 [0,1]。
    """
    if not in_01:
        I_gt = _to_01(I_gt)
        I_pred = _to_01(I_pred)
    out = {}
    out["psnr"] = psnr_batch(I_gt, I_pred, max_val=1.0).item()
    out["ssim"] = ssim_batch(I_gt, I_pred).item()
    if lpips_computer is not None:
        out["lpips"] = lpips_computer(I_gt, I_pred).item()
    else:
        out["lpips"] = None
    if fid_computer is not None:
        fr = fid_computer.get_features(I_gt)
        fp = fid_computer.get_features(I_pred)
        out["fid"] = FIDComputer.compute_from_features(fr, fp)
    else:
        out["fid"] = None
    return out


# ---------- 可选：clean-fid 对接（pip install clean-fid）----------
try:
    from clean_fid import fid
    _CLEAN_FID_AVAILABLE = True
except Exception:
    fid = None
    _CLEAN_FID_AVAILABLE = False


def compute_fid_cleanfid(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    device: torch.device,
    num_workers: int = 0,
) -> Optional[float]:
    """
    使用 clean-fid 计算 FID（若已安装）。
    real_imgs, fake_imgs: [B,3,H,W]，范围 [0,1]，将临时保存到目录后调用 clean_fid。
    """
    if not _CLEAN_FID_AVAILABLE or fid is None:
        return None
    import tempfile
    import shutil
    from PIL import Image
    def save_imgs(imgs: torch.Tensor, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        for i in range(imgs.shape[0]):
            x = (imgs[i].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(x).save(os.path.join(d, f"{i:05d}.png"))
    tmp = tempfile.mkdtemp()
    try:
        real_dir = os.path.join(tmp, "real")
        fake_dir = os.path.join(tmp, "fake")
        save_imgs(real_imgs, real_dir)
        save_imgs(fake_imgs, fake_dir)
        score = fid.compute_fid(real_dir, fake_dir, num_workers=num_workers)
        return float(score)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
