"""
统一验收入口：数据 / 模型 / 训练 / 指标 可选子项或全部。
用法：
  python scripts/verify.py data [--synthetic]
  python scripts/verify.py models
  python scripts/verify.py train
  python scripts/verify.py metrics
  python scripts/verify.py all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_data(args):
    import torch
    from torch.utils.data import DataLoader
    from data import SyntheticInpaintingDataset, build_dataset, make_fused_image
    from utils.config import get_config
    cfg = get_config(args.config)
    image_size = args.image_size or (cfg.get("data") or {}).get("image_size", 256)
    batch_size = args.batch_size or (cfg.get("train") or {}).get("batch_size", 4)
    mask_range = tuple((cfg.get("data") or {}).get("mask_range", [0.2, 0.8]))
    mask_type = (cfg.get("data") or {}).get("mask_type", "random")
    data_root = (cfg.get("data") or {}).get("root", "")
    use_synthetic = args.synthetic or not data_root or not os.path.isdir(os.path.expanduser(data_root))
    if use_synthetic:
        dataset = SyntheticInpaintingDataset(num_samples=32, image_size=image_size, mask_range=mask_range, mask_type=mask_type)
    else:
        dataset = build_dataset(name=(cfg.get("data") or {}).get("name", "celeba_hq"), root=data_root, image_size=image_size, mask_range=mask_range, mask_type=mask_type, train=True, train_split=(cfg.get("data") or {}).get("train_split", 0.95))
    if len(dataset) == 0:
        print("数据集为空，请使用 --synthetic 或正确 data.root"); sys.exit(1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    image, mask = batch["image"], batch["mask"]
    I_fus = make_fused_image(image, mask, noise_std=0.0)
    print("[data] image %s mask %s I_fus %s mask_mean %.4f" % (tuple(image.shape), tuple(mask.shape), tuple(I_fus.shape), mask.mean().item()))
    print("验收通过: 数据与 I_fus")


def cmd_models(args):
    import torch
    from models import (
        ChannelAttentionModule, SpatialAttentionModule, GraphAttentionModule,
        ContentAwareAttentionModule, AdaptiveAttentionModule, MultiAttentionFusion,
        MKAE, MultiScalePyramid, MixtureOfExperts, GatingNetwork, ExpertBlock,
        VelocityNetwork, RectifiedFlow, path_consistency_loss, sample_ode_euler, sample_ode_heun, sample_z_t,
        DoFe, UpFu, REME,
        MoEDiReF, LatentEncoder, LatentDecoder,
    )
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)
    for name, m in [("CAM", ChannelAttentionModule(C)), ("SAM", SpatialAttentionModule()), ("GAT", GraphAttentionModule(C)), ("CnAM", ContentAwareAttentionModule(C)), ("AdaAttn", AdaptiveAttentionModule(C)), ("MultiAttentionFusion", MultiAttentionFusion(C))]:
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == x.shape, name
    print("[models] attention ok")
    B, C_in = 2, 3
    x = torch.randn(B, C_in, H, W)
    mkae = MKAE(in_channels=C_in, base_channels=32, out_channels=64, num_experts=4)
    z_c, z_s = mkae(x, noise_std=0.0)
    assert z_c.shape == z_s.shape
    print("[models] MKAE ok")
    mkae_64 = MKAE(in_channels=64, base_channels=32, out_channels=64, num_experts=4)
    z_0, z_1 = torch.randn(B, 4, 16, 16), torch.randn(B, 4, 16, 16)
    t = torch.rand(B)
    rf = RectifiedFlow(latent_channels=4, channels=32, time_embed_dim=64, num_blocks=2)
    loss = rf.path_loss(z_0, z_1, t)
    z_0_samp = rf.sample(z_1, num_steps=5)
    assert z_0_samp.shape == z_1.shape
    print("[models] RectifiedFlow ok")
    mask = torch.rand(B, 1, H, W)
    reme = REME(dofe_in_channels=3, dofe_out_channels=64, latent_channels=4, mkae_out_channels=64)
    f_dofe = reme.forward_dofe(x, mask)
    z_c, z_s = mkae_64(f_dofe, noise_std=0.0)
    z_t = torch.randn(B, 4, 16, 16)
    z_ct = reme.constrain_latent(z_t, z_c, z_s)
    assert z_ct.shape == z_t.shape
    print("[models] REME ok")
    model = MoEDiReF(in_channels=3, latent_channels=64, dofe_out_channels=64, mkae_out_channels=64, num_experts=4)
    I_gt, I_fus = torch.randn(B, 3, 64, 64), torch.randn(B, 3, 64, 64)
    with torch.no_grad():
        out = model(I_gt, I_fus, mask, return_loss_components=True, num_steps=5)
    assert out["I_pred"].shape == I_gt.shape and out["loss_path"].dim() == 0
    I_infer = model.infer(I_fus, mask, num_steps=5)
    assert I_infer.shape == I_fus.shape
    print("[models] MoEDiReF ok")
    print("验收通过: 所有模型前向")


def cmd_train(args):
    import torch
    from torch.utils.data import DataLoader
    from utils.config import get_config
    from data import SyntheticInpaintingDataset, make_fused_image
    from models import MoEDiReF
    from losses import reconstruction_loss, perceptual_loss, VGG16PerceptualLoss, PatchDiscriminator, wgan_gp_d_loss, wgan_gp_g_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config(args.config)
    train_cfg = cfg.get("train", {})
    lambda_rec = train_cfg.get("lambda_rec", 1.0)
    lambda_perc = train_cfg.get("lambda_perc", 0.1)
    lambda_adv = train_cfg.get("lambda_adv", 0.1)
    lambda_path = train_cfg.get("lambda_path", 1.0)
    lambda_gp = train_cfg.get("lambda_gp", 10.0)
    B, image_size = 2, 64
    dataset = SyntheticInpaintingDataset(num_samples=8, image_size=image_size)
    loader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)
    model = MoEDiReF(in_channels=3, latent_channels=32, dofe_out_channels=32, mkae_out_channels=32, num_experts=2).to(device)
    D = PatchDiscriminator(in_channels=3, ndf=32, n_layers=2).to(device)
    try:
        vgg = VGG16PerceptualLoss(weight_initialized=False).to(device).eval()
    except Exception:
        vgg = None
    opt_g = torch.optim.Adam(model.parameters(), lr=2e-4)
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4)
    model.train(); D.train()
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        I_fus = make_fused_image(image, mask, noise_std=0.0)
        out = model(image, I_fus, mask, return_loss_components=True, num_steps=5)
        I_pred, loss_path = out["I_pred"], out["loss_path"]
        L_rec = reconstruction_loss(image, I_pred)
        I_gt_vgg = torch.clamp((image + 1) * 0.5, 0, 1)
        I_pred_vgg = torch.clamp((I_pred + 1) * 0.5, 0, 1)
        L_perc = perceptual_loss(I_gt_vgg, I_pred_vgg, vgg) if vgg is not None else torch.tensor(0.0, device=device)
        L_d = wgan_gp_d_loss(D, image, I_pred.detach(), lambda_gp=lambda_gp)
        opt_d.zero_grad(); L_d.backward(); opt_d.step()
        L_g_adv = wgan_gp_g_loss(D, I_pred)
        L_total = lambda_rec * L_rec + lambda_perc * L_perc + lambda_adv * L_g_adv + lambda_path * loss_path
        opt_g.zero_grad(); L_total.backward(); opt_g.step()
        if i == 0:
            print("[train] L_rec %.4f L_perc %.4f loss_path %.4f L_d %.4f L_total %.4f" % (L_rec.item(), L_perc.item(), loss_path.item(), L_d.item(), L_total.item()))
    print("验收通过: 训练循环")


def cmd_metrics(args):
    import torch
    from torch.utils.data import DataLoader
    from utils.metrics import psnr_batch, ssim_batch, compute_metrics, LPIPSComputer, _to_01
    from data import SyntheticInpaintingDataset, make_fused_image
    from models import MoEDiReF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 2, 64, 64
    I_gt = torch.rand(B, 3, H, W, device=device)
    I_pred = I_gt + 0.1 * torch.randn_like(I_gt, device=device)
    I_pred = torch.clamp(I_pred, 0, 1)
    print("[metrics] PSNR %.4f SSIM %.4f" % (psnr_batch(I_gt, I_pred, max_val=1.0).item(), ssim_batch(I_gt, I_pred).item()))
    out = compute_metrics(I_gt, I_pred, in_01=True, lpips_computer=LPIPSComputer(device))
    print("[metrics] compute_metrics psnr=%.4f ssim=%.4f lpips=%s" % (out["psnr"], out["ssim"], out["lpips"]))
    dataset = SyntheticInpaintingDataset(num_samples=8, image_size=64)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MoEDiReF(in_channels=3, latent_channels=32, dofe_out_channels=32, mkae_out_channels=32, num_experts=2).to(device).eval()
    batch = next(iter(loader))
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    I_fus = make_fused_image(image, mask)
    with torch.no_grad():
        I_pred_infer = model.infer(I_fus, mask, num_steps=4, solver="euler")
    I_pred_infer = torch.clamp(I_pred_infer, -1, 1)
    p = psnr_batch(_to_01(image), _to_01(I_pred_infer), max_val=1.0).item()
    s = ssim_batch(_to_01(image), _to_01(I_pred_infer)).item()
    print("[metrics] inference PSNR=%.4f SSIM=%.4f" % (p, s))
    print("验收通过: 指标与推理流程")


def main():
    parser = argparse.ArgumentParser(description="MoE-DiReF 统一验收")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="仅 data 子命令有效：使用合成数据")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data", help="数据加载与 I_fus")
    sub.add_parser("models", help="各模型前向")
    sub.add_parser("train", help="训练若干 step")
    sub.add_parser("metrics", help="指标与推理")
    p_all = sub.add_parser("all", help="依次执行 data -> models -> train -> metrics")
    args = parser.parse_args()
    if args.cmd == "data":
        cmd_data(args)
    elif args.cmd == "models":
        cmd_models(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "metrics":
        cmd_metrics(args)
    elif args.cmd == "all":
        cmd_data(args)
        print()
        cmd_models(args)
        print()
        cmd_train(args)
        print()
        cmd_metrics(args)
        print("\n全部验收通过.")


if __name__ == "__main__":
    main()
