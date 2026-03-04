"""
MoE-DiReF 训练入口：L_rec + L_perc + L_adv (WGAN-GP) + L_path，cosine 学习率，论文 III-E。
"""

import os
import sys
import argparse
import csv
from datetime import datetime

# 确保项目根目录在 Python 路径中，避免 ModuleNotFoundError
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.config import get_config
from data import build_dataset, make_fused_image, SyntheticInpaintingDataset
from models import MoEDiReF
from losses import (
    reconstruction_loss,
    perceptual_loss,
    VGG16PerceptualLoss,
    PatchDiscriminator,
    wgan_gp_d_loss,
    wgan_gp_g_loss,
)


def _to_vgg_range(x: torch.Tensor) -> torch.Tensor:
    """将 [-1, 1] 转到 [0, 1] 供 VGG 使用。"""
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="用合成数据快速跑通")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = get_config(args.config, args.override)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    image_size = args.image_size or data_cfg.get("image_size", 256)
    batch_size = args.batch_size or train_cfg.get("batch_size", 16)
    max_epochs = args.max_epochs or train_cfg.get("epochs", 2000)
    lambda_rec = train_cfg.get("lambda_rec", 1.0)
    lambda_perc = train_cfg.get("lambda_perc", 0.1)
    lambda_adv = train_cfg.get("lambda_adv", 0.1)
    lambda_path = train_cfg.get("lambda_path", 1.0)
    lambda_gp = train_cfg.get("lambda_gp", 10.0)
    lr_max = train_cfg.get("lr_max", 2e-4)
    lr_min = train_cfg.get("lr_min", 5e-6)
    beta1 = train_cfg.get("beta1", 0.9)
    beta2 = train_cfg.get("beta2", 0.999)
    save_every = train_cfg.get("save_every", 100)
    log_every = train_cfg.get("log_every", 10)
    rf_steps_train = train_cfg.get("rf_steps_train", 10)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # 日志：CSV + TensorBoard
    log_csv_path = os.path.join(args.save_dir, "train_log.csv")
    is_new_log = not os.path.exists(log_csv_path)
    log_file = open(log_csv_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if is_new_log:
        log_writer.writerow(
            [
                "step",
                "epoch",
                "rec",
                "perc",
                "adv",
                "path",
                "total",
                "lr_g",
                "lr_d",
            ]
        )

    tb_log_dir = os.path.join(args.save_dir, "tb")
    writer = SummaryWriter(log_dir=tb_log_dir)
    writer.add_text(
        "config",
        f"config={args.config}, override={args.override}, synthetic={args.synthetic}, "
        f"batch_size={batch_size}, image_size={image_size}",
    )

    if args.synthetic:
        dataset = SyntheticInpaintingDataset(
            num_samples=256,
            image_size=image_size,
            mask_range=tuple(data_cfg.get("mask_range", [0.2, 0.8])),
            mask_type=data_cfg.get("mask_type", "random"),
        )
    else:
        dataset = build_dataset(
            name=data_cfg.get("name", "celeba_hq"),
            root=data_cfg.get("root", "./data"),
            image_size=image_size,
            mask_range=tuple(data_cfg.get("mask_range", [0.2, 0.8])),
            mask_type=data_cfg.get("mask_type", "random"),
            train=True,
            train_split=data_cfg.get("train_split", 0.95),
        )
    if len(dataset) == 0:
        print("数据集为空，请使用 --synthetic 或指定正确 data.root")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    latent_channels = model_cfg.get("latent_dim", 64)
    in_channels = model_cfg.get("in_channels", 3)
    model = MoEDiReF(
        in_channels=in_channels,
        latent_channels=latent_channels,
        dofe_out_channels=latent_channels,
        mkae_out_channels=latent_channels,
        num_experts=model_cfg.get("num_experts", 4),
    ).to(device)
    discriminator = PatchDiscriminator(in_channels=in_channels).to(device)
    try:
        vgg = VGG16PerceptualLoss(weight_initialized=True).to(device)
        vgg.eval()
    except Exception:
        vgg = None

    opt_g = torch.optim.Adam(model.parameters(), lr=lr_max, betas=(beta1, beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_max, betas=(beta1, beta2))
    total_steps = len(loader) * max_epochs
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=total_steps, eta_min=lr_min)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=total_steps, eta_min=lr_min)

    global_step = 0
    for epoch in range(max_epochs):
        model.train()
        discriminator.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for batch in pbar:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            I_fus = make_fused_image(image, mask, noise_std=0.0)

            out = model(image, I_fus, mask, return_loss_components=True, num_steps=rf_steps_train)
            I_pred = out["I_pred"]
            loss_path = out["loss_path"]

            L_rec = reconstruction_loss(image, I_pred)
            if vgg is not None:
                L_perc = perceptual_loss(
                    _to_vgg_range(image),
                    _to_vgg_range(torch.clamp(I_pred, -1, 1)),
                    vgg,
                )
            else:
                L_perc = torch.tensor(0.0, device=device)
            L_d = wgan_gp_d_loss(discriminator, image, I_pred.detach(), lambda_gp=lambda_gp)
            opt_d.zero_grad()
            L_d.backward()
            opt_d.step()

            L_g_adv = wgan_gp_g_loss(discriminator, I_pred)
            L_total = lambda_rec * L_rec + lambda_perc * L_perc + lambda_adv * L_g_adv + lambda_path * loss_path
            opt_g.zero_grad()
            L_total.backward()
            opt_g.step()

            sched_g.step()
            sched_d.step()
            global_step += 1

            if global_step % log_every == 0:
                lr_g = opt_g.param_groups[0]["lr"]
                lr_d = opt_d.param_groups[0]["lr"]
                log_writer.writerow(
                    [
                        global_step,
                        epoch + 1,
                        L_rec.item(),
                        L_perc.item(),
                        L_g_adv.item(),
                        loss_path.item(),
                        L_total.item(),
                        lr_g,
                        lr_d,
                    ]
                )
                log_file.flush()

                writer.add_scalar("loss/rec", L_rec.item(), global_step)
                writer.add_scalar("loss/perc", L_perc.item(), global_step)
                writer.add_scalar("loss/adv", L_g_adv.item(), global_step)
                writer.add_scalar("loss/path", loss_path.item(), global_step)
                writer.add_scalar("loss/total", L_total.item(), global_step)
                writer.add_scalar("lr/g", lr_g, global_step)
                writer.add_scalar("lr/d", lr_d, global_step)

            pbar.set_postfix(
                rec=f"{L_rec.item():.4f}",
                path=f"{loss_path.item():.4f}",
                d=f"{L_d.item():.4f}",
            )
        if (epoch + 1) % save_every == 0 or epoch == 0:
            path = os.path.join(args.save_dir, f"moe_diref_ep{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
            }, path)
            print(f"Saved {path}")
    log_file.close()
    writer.close()
    print("Training done.")


if __name__ == "__main__":
    main()
