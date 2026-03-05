"""
单张图片推理：指定图像与 mask，输出修复结果。
用法：
  python scripts/infer_single.py --checkpoint ckpt.pt --image img.png --mask mask.png --output out.png
  python scripts/infer_single.py --checkpoint ckpt.pt --image img.png --output out.png  # 自动生成随机 mask
  python scripts/infer_single.py --checkpoint ckpt.pt --image img.png --save_diagram ./diagram  # 保存图示四件套
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


def _tensor_to_uint8_rgb(x: torch.Tensor, h: int = None, w: int = None) -> np.ndarray:
    """[-1,1] [B,C,H,W] -> [H,W,3] uint8，可选 resize 到 (h,w)。"""
    img = (x[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if h is not None and w is not None and (img.shape[0], img.shape[1]) != (h, w):
        img = np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None, help="二值 mask 图，白(255)为损坏区；不提供则自动生成随机 mask")
    parser.add_argument("--output", type=str, default=None, help="修复结果保存路径，默认 image 同目录下 _inpaint.png")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="未提供 mask 时随机 mask 的缺失率（默认 25%%）")
    parser.add_argument("--mask_type", type=str, default="irregular", choices=["random", "rectangle", "irregular"], help="未提供 mask 时的形状：irregular=不规则，rectangle=矩形，random=随机二选一")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_diagram", type=str, default=None, help="保存图示四件套的目录：损坏输入、二值掩码、噪声潜码、恢复高分辨率输出")
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

    mask_np_orig = None  # 仅自动生成 mask 时为原图尺寸，用于图示
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
        # 按原图尺寸生成 mask，再下采样到模型尺寸，使缺失区域随图片大小自适应
        h_orig, w_orig = img.height, img.width
        mask_np = random_mask(
            h_orig,
            w_orig,
            mask_ratio_range=(args.mask_ratio, args.mask_ratio),
            mask_type=args.mask_type,
        )
        mask_np_orig = mask_np.astype(np.float32)  # 保留原图尺寸，供图示等使用
        mask_pil_small = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (image_size, image_size), Image.NEAREST
        )
        mask_np_small = np.array(mask_pil_small).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask_np_small).unsqueeze(0).unsqueeze(0).float().to(device)

    I_fus = make_fused_image(I_gt, mask, noise_std=0.0)
    model = load_model(args.checkpoint, device, cfg)

    if args.save_diagram:
        # 生成图示四件套：损坏输入、二值掩码、噪声潜码、恢复高分辨率输出
        with torch.no_grad():
            out_dict = model.infer_with_intermediates(
                I_fus, mask, num_steps=args.num_steps, solver=args.solver
            )
        I_pred = torch.clamp(out_dict["I_pred"], -1.0, 1.0)
        noise_decoded = torch.clamp(out_dict["noise_decoded"], -1.0, 1.0)

        os.makedirs(args.save_diagram, exist_ok=True)
        h, w = img.height, img.width

        # 1. 损坏输入 (Damaged Image)：缺失区在模型里是 0（[-1,1]）→ 转成图会变灰，图示里改为黑更直观
        damaged = _tensor_to_uint8_rgb(I_fus, h, w)
        if mask_np_orig is not None:
            mask_vis = (mask_np_orig > 0.5).astype(np.uint8)
        else:
            m = mask[0, 0].cpu().numpy()
            m_pil = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
            mask_vis = (np.array(m_pil) > 127).astype(np.uint8)
        damaged[mask_vis > 0] = 0  # 缺失区图示为黑色
        Image.fromarray(damaged).save(os.path.join(args.save_diagram, "1_damaged_input.png"))

        # 2. 二值掩码 (Binary Mask)：损坏=黑、保留=白，严格 0/255 避免灰边
        if mask_np_orig is not None:
            m_uint8 = (mask_np_orig > 0.5).astype(np.uint8) * 255
        else:
            m = mask[0, 0].cpu().numpy()
            m_uint8 = (m > 0.5).astype(np.uint8) * 255
            if (m_uint8.shape[0], m_uint8.shape[1]) != (h, w):
                m_uint8 = np.array(Image.fromarray(m_uint8).resize((w, h), Image.NEAREST))
        m_uint8 = 255 - m_uint8  # 显示：损坏=黑(0)、保留=白(255)
        mask_rgb = np.stack([m_uint8, m_uint8, m_uint8], axis=-1)
        Image.fromarray(mask_rgb).save(os.path.join(args.save_diagram, "2_binary_mask.png"))

        # 3. 噪声潜码 (Noise Latent)：解码 z_1 得到的可视化
        noise_img = _tensor_to_uint8_rgb(noise_decoded, h, w)
        Image.fromarray(noise_img).save(os.path.join(args.save_diagram, "3_noise_latent_decoded.png"))

        # 4. 恢复高分辨率输出 (Restored High-Res Image)
        restored = _tensor_to_uint8_rgb(I_pred, h, w)
        Image.fromarray(restored).save(os.path.join(args.save_diagram, "4_restored_highres.png"))

        print("Diagram outputs saved to", args.save_diagram)
        print("  1_damaged_input.png, 2_binary_mask.png, 3_noise_latent_decoded.png, 4_restored_highres.png")
    else:
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
