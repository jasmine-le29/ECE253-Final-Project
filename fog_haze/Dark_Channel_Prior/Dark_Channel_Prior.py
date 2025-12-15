#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dark Channel Prior 去雾（Python 3.10 兼容版）
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 新增：只为 LPIPS 准备的依赖 ===
import torch
import lpips


def cal_Dark_Channel(im: np.ndarray, width: int = 15) -> np.ndarray:
    """
    计算暗通道
    im: 输入彩色图，float32/float64，范围 [0, 1]，形状 (H, W, 3)
    width: 窗口大小（奇数）
    """
    # 每个像素在 RGB 通道取最小
    im_dark = np.min(im, axis=2)

    border = (width - 1) // 2
    im_dark_1 = cv2.copyMakeBorder(
        im_dark,
        border,
        border,
        border,
        border,
        borderType=cv2.BORDER_DEFAULT,
    )

    h, w = im_dark.shape
    res = np.zeros_like(im_dark)

    # 逐像素滑动窗口取最小值（可以工作，速度一般；如果想更快可用 erode）
    for i in range(h):
        for j in range(w):
            window = im_dark_1[i:i + width, j:j + width]
            res[i, j] = np.min(window)

    return res


def cal_Light_A(dark: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    估计大气光 A
    dark: 暗通道图像 (H, W)
    img: 原图 (H, W, 3)，float [0, 1]
    """
    h, w = dark.shape
    num_pixels = h * w
    num = max(int(num_pixels * 0.001), 1)  # 取 top 0.1% 的暗通道像素

    # 展平后取暗通道最大的若干位置
    dark_vec = dark.reshape(-1)
    img_vec = img.reshape(-1, 3)

    indices = np.argsort(dark_vec)[-num:]  # 从小到大排，取最后 num 个

    # 在这些位置中，取 RGB 最亮的像素作为 A
    brightest = img_vec[indices]
    A = brightest.max(axis=0)

    return A


def harz_Rec(A: np.ndarray, img: np.ndarray, t: np.ndarray, t0: float = 0.1) -> np.ndarray:
    """
    根据大气散射模型恢复无雾图像
    A: 大气光 (3,)
    img: 原图 (H, W, 3)，float [0, 1]
    t: 透射率图 (H, W)
    t0: 透射率下限
    """
    t_clamped = np.maximum(t, t0)  # 防止过小

    # 广播到每个通道
    J = (img - A) / t_clamped[..., np.newaxis] + A
    J = np.clip(J, 0.0, 1.0)
    return J


def cal_trans(A: np.ndarray, img: np.ndarray, w: float = 0.95) -> np.ndarray:
    """
    计算粗略透射率
    A: 大气光 (3,)
    img: 原图 (H, W, 3)，float [0, 1]
    w: omega 参数
    """
    # I / A
    normed = img / A
    dark = cal_Dark_Channel(normed)
    t = np.maximum(1 - w * dark, 0.0)
    return t


def Guided_filtering(t: np.ndarray,
                     img_gray: np.ndarray,
                     width: int,
                     sigma: float = 1e-4) -> np.ndarray:
    """
    单通道引导滤波（与原代码一致）
    t: 粗略透射率 (H, W)，float
    img_gray: 灰度引导图 (H, W)，float [0, 1]
    width: 方窗大小
    sigma: 正则项
    """
    ksize = (width, width)

    mean_I = cv2.boxFilter(img_gray, ddepth=-1, ksize=ksize,
                           normalize=True, borderType=cv2.BORDER_DEFAULT)
    mean_t = cv2.boxFilter(t, ddepth=-1, ksize=ksize,
                           normalize=True, borderType=cv2.BORDER_DEFAULT)

    corr_I = cv2.boxFilter(img_gray * img_gray, ddepth=-1, ksize=ksize,
                           normalize=True, borderType=cv2.BORDER_DEFAULT)
    corr_IT = cv2.boxFilter(img_gray * t, ddepth=-1, ksize=ksize,
                            normalize=True, borderType=cv2.BORDER_DEFAULT)

    var_I = corr_I - mean_I * mean_I
    cov_IT = corr_IT - mean_I * mean_t

    a = cov_IT / (var_I + sigma)
    b = mean_t - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize,
                           normalize=True, borderType=cv2.BORDER_DEFAULT)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize,
                           normalize=True, borderType=cv2.BORDER_DEFAULT)

    q = mean_a * img_gray + mean_b
    return q


if __name__ == "__main__":
    # 读入有雾图像
    img_bgr = cv2.imread("foggy.jpg")
    if img_bgr is None:
        raise FileNotFoundError("找不到，请检查路径")

    # 转 float32 并归一化到 [0,1]
    print(img_bgr.shape)
    img = img_bgr.astype(np.float32) / 255.0

    # 用同一张图转换成灰度图做引导
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 显示原图（注意 Matplotlib 用 RGB）
    plt.figure()
    plt.title("Hazy Image")
    plt.imshow(img[:, :, ::-1])  # BGR -> RGB
    plt.axis("off")
    plt.show()

    # 暗通道
    im_dark = cal_Dark_Channel(img)
    plt.figure()
    plt.title("Dark Channel")
    plt.imshow(im_dark, cmap="gray")
    plt.axis("off")
    plt.show()

    # 大气光
    A = cal_Light_A(im_dark, img)
    print("Estimated A:", A)

    # 粗略透射率
    trans = cal_trans(A, img)
    plt.figure()
    plt.title("Transmission (raw)")
    plt.imshow(trans, cmap="gray")
    plt.axis("off")
    plt.show()

    # 引导滤波细化透射率
    trans_refined = Guided_filtering(trans, img_gray, width=41)
    plt.figure()
    plt.title("Transmission (refined)")
    plt.imshow(trans_refined, cmap="gray")
    plt.axis("off")
    plt.show()

    # 恢复无雾图像
    result = harz_Rec(A, img, trans_refined)

    # 显示与保存
    plt.figure()
    plt.title("Dehazed Result")
    plt.imshow(result[:, :, ::-1])  # BGR -> RGB
    plt.axis("off")
    plt.show()

    result_uint8 = (result * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite("dehazed_img.png", result_uint8)
    print("去雾结果已保存为 dehazed_img.png")

    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import torch
    import lpips

    # ======== 读入 GT 图像 38_gt.png ========
    gt_bgr = cv2.imread("38_gt.png")
    if gt_bgr is None:
        raise FileNotFoundError("找不到 gt图像，请检查路径")

    # 确保 GT 和 result 尺寸一致
    gt_bgr = cv2.resize(gt_bgr, (result_uint8.shape[1], result_uint8.shape[0]))

    # 转为 RGB + float32 + [0,1]
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    dehazed_rgb = cv2.cvtColor(result_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ======== 计算 PSNR ========
    psnr_val = psnr(gt_rgb, dehazed_rgb, data_range=1.0)
    print(f"PSNR: {psnr_val:.4f} dB")

    # ======== 计算 SSIM（多通道）========
    ssim_val = ssim(gt_rgb, dehazed_rgb, channel_axis=2, data_range=1.0)
    print(f"SSIM: {ssim_val:.4f}")

    # ======== LPIPS - 使用 AlexNet Backbone ========
    loss_fn = lpips.LPIPS(net='alex')  # 也可改为 'vgg'
    loss_fn.eval()

    # [0,1] → [-1,1] 并转为 PyTorch Tensor
    gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    res_t = torch.from_numpy(dehazed_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    with torch.no_grad():
        lpips_val = loss_fn(gt_t, res_t).item()

    print(f"LPIPS: {lpips_val:.4f}")

