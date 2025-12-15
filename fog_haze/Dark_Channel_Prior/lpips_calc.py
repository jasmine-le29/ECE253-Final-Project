#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import lpips
from PIL import Image


def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    """
    读取图片 -> RGB -> [0,1] -> (1,3,H,W) -> [-1,1] 的 torch.Tensor
    """
    img = Image.open(path).convert("RGB")
    img = img.copy()  # 避免某些 lazy 读取的问题

    # 转成 [0,1] 的 float32
    import numpy as np
    img_np = (np.array(img).astype("float32") / 255.0)  # (H,W,3)

    # [H,W,3] -> [1,3,H,W]
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # [0,1] -> [-1,1]
    img_t = img_t * 2.0 - 1.0

    return img_t.to(device)


def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS between two images.")
    parser.add_argument("img0", type=str, help="Path to first image (reference or GT).")
    parser.add_argument("img1", type=str, help="Path to second image (e.g., result).")
    parser.add_argument(
        "--net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone for LPIPS (default: alex)",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize img1 to img0's size before computing LPIPS.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force use CPU even if CUDA is available.",
    )

    args = parser.parse_args()

    # 设备
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 先用 PIL 打开，做尺寸对齐（如果需要）
    img0_path = Path(args.img0)
    img1_path = Path(args.img1)

    if not img0_path.is_file():
        raise FileNotFoundError(f"img0 not found: {img0_path}")
    if not img1_path.is_file():
        raise FileNotFoundError(f"img1 not found: {img1_path}")

    from PIL import Image

    img0_pil = Image.open(img0_path).convert("RGB")
    img1_pil = Image.open(img1_path).convert("RGB")

    if args.resize:
        img1_pil = img1_pil.resize(img0_pil.size, Image.BICUBIC)

    # 临时保存到内存，再用前面的函数转 tensor
    import numpy as np
    img0_np = (np.array(img0_pil).astype("float32") / 255.0)
    img1_np = (np.array(img1_pil).astype("float32") / 255.0)

    img0_t = torch.from_numpy(img0_np).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    img1_t = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    img0_t = img0_t.to(device)
    img1_t = img1_t.to(device)

    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net=args.net).to(device)
    loss_fn.eval()

    with torch.no_grad():
        dist = loss_fn(img0_t, img1_t).item()

    print(f"LPIPS ({args.net}) between:\n  {img0_path}\n  {img1_path}\n= {dist:.6f}")


if __name__ == "__main__":
    main()
