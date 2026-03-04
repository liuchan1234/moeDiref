"""
单张图片推理：指定图像与 mask，输出修复结果。
用法：
  python scripts/infer_single.py --checkpoint ckpt.pt --image img.png --mask mask.png --output out.png
  python scripts/infer_single.py --checkpoint ckpt.pt --image img.png --output out.png  # 自动生成随机 mask
"""

import os
import sys
import argparse

import torch
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from inference import load_model
from data.transforms import get_eval_transforms
from data.mask import random_mask, make_fused_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None, help="二值 mask 图，白(255)为损坏区；不提供则自动生成随机 mask")
    parser.add_argument("--output", type=str, default=None, help="修复结果保存路径，默认 image 同目录下 _inpaint.png")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="未提供 mask 时随机 mask 的缺失率")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = get_config(args.config)
    image_size = cfg.get("data", {}).get("image_size", 256)
    transform = get_eval_transforms(image_size, normalize=True)

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    I_gt = transform(img)
    if I_gt.dim() == 2:
        I_gt = I_gt.unsqueeze(0)
    I_gt = I_gt.unsqueeze(0).to(device)

    if args.mask and os.path.isfile(args.mask):
        mask_pil = Image.open(args.mask).convert("L")
        mask_np = np.array(mask_pil)
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.float32)
        else:
            mask_np = mask_np.astype(np.float32)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((image_size, image_size), Image.NEAREST)
        mask_np = np.array(mask_resized).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    else:
        mask_np = random_mask(image_size, image_size, mask_ratio_range=(args.mask_ratio, args.mask_ratio))
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

    I_fus = make_fused_image(I_gt, mask, noise_std=0.0)
    model = load_model(args.checkpoint, device, cfg)
    with torch.no_grad():
        I_pred = model.infer(I_fus, mask, num_steps=args.num_steps, solver=args.solver)
    I_pred = torch.clamp(I_pred, -1.0, 1.0)
    out = ((I_pred[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out)
    if out_pil.size != (img.width, img.height):
        out_pil = out_pil.resize((img.width, img.height), Image.BILINEAR)

    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(args.image)
        out_path = base + "_inpaint.png"
    out_pil.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
