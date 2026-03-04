"""
MoE-DiReF 推理与评估：加载 checkpoint、跑测试集、计算 PSNR/SSIM/FID/LPIPS、保存可视化。
"""

import os
import sys
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import get_config
from utils.metrics import (
    psnr_batch,
    ssim_batch,
    FIDComputer,
    LPIPSComputer,
    compute_metrics,
    compute_fid_cleanfid,
    _to_01,
)
from data import build_dataset, make_fused_image, SyntheticInpaintingDataset
from models import MoEDiReF


def load_model(
    checkpoint_path: str,
    device: torch.device,
    config: Optional[dict] = None,
) -> MoEDiReF:
    """从 checkpoint 加载 MoEDiReF（仅恢复 model 权重）。"""
    cfg = config or get_config()
    model_cfg = cfg.get("model", {})
    model = MoEDiReF(
        in_channels=cfg.get("model", {}).get("in_channels", 3),
        latent_channels=model_cfg.get("latent_dim", 64),
        dofe_out_channels=model_cfg.get("latent_dim", 64),
        mkae_out_channels=model_cfg.get("latent_dim", 64),
        num_experts=model_cfg.get("num_experts", 4),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run_eval(
    model: MoEDiReF,
    loader: DataLoader,
    device: torch.device,
    num_steps: int = 10,
    solver: str = "euler",
    compute_fid: bool = False,
    use_clean_fid: bool = False,
    compute_lpips: bool = True,
    save_dir: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> dict:
    """
    在 loader 上推理并汇总 PSNR、SSIM，可选 FID、LPIPS。
    use_clean_fid：若 True 且已安装 clean-fid，则用 clean-fid 算 FID。
    若 save_dir 非空，保存前若干张：I_gt, I_fus, I_pred 拼接图。
    """
    fid_computer = FIDComputer(device) if (compute_fid and not use_clean_fid) else None
    lpips_computer = LPIPSComputer(device) if compute_lpips else None

    psnr_list = []
    ssim_list = []
    lpips_list = []
    all_gt = []
    all_pred = []

    total = min(max_batches or len(loader), len(loader))
    for bi, batch in tqdm(enumerate(loader), total=total, desc="Eval"):
        if max_batches is not None and bi >= max_batches:
            break
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        I_fus = make_fused_image(image, mask, noise_std=0.0)

        I_pred = model.infer(I_fus, mask, num_steps=num_steps, solver=solver)
        I_pred = torch.clamp(I_pred, -1.0, 1.0)

        I_gt_01 = _to_01(image)
        I_pred_01 = _to_01(I_pred)
        psnr_list.append(psnr_batch(I_gt_01, I_pred_01, max_val=1.0).item())
        ssim_list.append(ssim_batch(I_gt_01, I_pred_01).item())
        if lpips_computer is not None:
            lpips_list.append(lpips_computer(I_gt_01, I_pred_01).item())
        if compute_fid:
            all_gt.append(I_gt_01)
            all_pred.append(I_pred_01)

        if save_dir and bi < 20:
            os.makedirs(save_dir, exist_ok=True)
            for i in range(image.shape[0]):
                if bi * loader.batch_size + i >= 20:
                    break
                idx = bi * loader.batch_size + i
                gt = I_gt_01[i].cpu().permute(1, 2, 0).numpy()
                fus = _to_01(I_fus[i]).cpu().permute(1, 2, 0).numpy()
                pred = I_pred_01[i].cpu().permute(1, 2, 0).numpy()
                try:
                    import numpy as np
                    from PIL import Image
                    row = np.concatenate([gt, fus, pred], axis=1)
                    row = (row * 255).clip(0, 255).astype("uint8")
                    Image.fromarray(row).save(os.path.join(save_dir, f"vis_{idx:04d}.png"))
                except Exception:
                    pass

    results = {
        "psnr": sum(psnr_list) / len(psnr_list) if psnr_list else 0.0,
        "ssim": sum(ssim_list) / len(ssim_list) if ssim_list else 0.0,
        "lpips": sum(lpips_list) / len(lpips_list) if lpips_list else None,
        "fid": None,
    }
    if compute_fid and all_gt and all_pred:
        all_gt = torch.cat(all_gt, dim=0)
        all_pred = torch.cat(all_pred, dim=0)
        if use_clean_fid:
            results["fid"] = compute_fid_cleanfid(all_gt, all_pred, device)
        elif fid_computer is not None:
            results["fid"] = FIDComputer.compute_from_features(
                fid_computer.get_features(all_gt),
                fid_computer.get_features(all_pred),
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--save_vis", type=str, default=None, help="可视化保存目录")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--no_fid", action="store_true")
    parser.add_argument("--use_clean_fid", action="store_true", help="用 clean-fid 算 FID（需 pip install clean-fid）")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = cfg.get("data", {})
    image_size = args.image_size or data_cfg.get("image_size", 256)

    model = load_model(args.checkpoint, device, cfg)

    if args.synthetic:
        dataset = SyntheticInpaintingDataset(
            num_samples=min(64, args.max_batches * args.batch_size if args.max_batches else 64),
            image_size=image_size,
            mask_range=tuple(data_cfg.get("mask_range", [0.2, 0.8])),
        )
    else:
        root = args.data_root or data_cfg.get("root", "./data")
        dataset = build_dataset(
            name=data_cfg.get("name", "celeba_hq"),
            root=root,
            image_size=image_size,
            mask_range=tuple(data_cfg.get("mask_range", [0.2, 0.8])),
            train=False,
            train_split=data_cfg.get("train_split", 0.95),
        )
    if len(dataset) == 0:
        print("测试集为空，使用 --synthetic 或指定 --data_root")
        dataset = SyntheticInpaintingDataset(num_samples=16, image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    results = run_eval(
        model,
        loader,
        device,
        num_steps=args.num_steps,
        solver=args.solver,
        compute_fid=not args.no_fid,
        use_clean_fid=args.use_clean_fid,
        compute_lpips=not args.no_lpips,
        save_dir=args.save_vis,
        max_batches=args.max_batches,
    )

    print("Evaluation results:")
    print(f"  PSNR  (↑): {results['psnr']:.4f}")
    print(f"  SSIM  (↑): {results['ssim']:.4f}")
    if results["lpips"] is not None:
        print(f"  LPIPS (↓): {results['lpips']:.4f}")
    if results["fid"] is not None:
        print(f"  FID   (↓): {results['fid']:.4f}")
    if args.save_vis:
        print(f"  Visualizations saved to {args.save_vis}")


if __name__ == "__main__":
    main()
