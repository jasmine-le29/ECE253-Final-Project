import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/Dehaze_NH.json',
        help='JSON file for evaluation configuration'
    )
    parser.add_argument(
        '-gpu', '--gpu_ids',
        type=str,
        default=None,
        help='GPU ids: e.g. "0" or "0,1"; leave None for CPU'
    )
    parser.add_argument(
        '-debug', '-d',
        action='store_true',
        help='Debug mode'
    )
    parser.add_argument(
        "-p", "--phase", 
        type=str, 
        choices=["train", "val"], 
        default="val",
        help="Run either train or val"
    )
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()

    # 解析 config，复用你原来的 Logger.parse 逻辑
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # 强制 phase 为 'val'，避免配置里写错
    opt['phase'] = 'val'

    # cuDNN 设置
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # 日志
    Logger.setup_logger(None, opt['path']['log'], 'eval', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info('**************** Evaluation Config ****************')
    logger.info(Logger.dict2str(opt))
    logger.info('***************************************************')

    # ====== 数据集：只用 val 部分 ======
    dataset_opt = opt['datasets']['val']
    val_set = Data.create_dataset(dataset_opt, phase='val')
    val_loader = Data.create_dataloader(val_set, dataset_opt, phase='val')
    logger.info('Initial Val Dataset Finished, length: {}'.format(len(val_set)))

    # ====== 模型 ======
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = getattr(diffusion, 'begin_step', 0)
    current_epoch = getattr(diffusion, 'begin_epoch', 0)

    # 使用验证用的 noise schedule
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'],
        schedule_phase='val'
    )

    # ====== 评估循环 ======
    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0

    # 结果保存目录：results/epoch_xxx 或直接 results/
    result_root = opt['path']['results']
    os.makedirs(result_root, exist_ok=True)

    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)

        # continous=True：生成中间过程 SR 序列（和你原来 eval 分支保持一致）
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        # 取出 HR / 生成结果
        hr_img = Metrics.tensor2img(visuals['HR'])           # uint8, GT
        # SR 是一个时间序列，最后一帧是最终去雾结果
        sr_seq = visuals['SR']                               # Tensor [T, C, H, W]
        sr_last = Metrics.tensor2img(sr_seq[-1])             # 最后一帧作为输出

        # 如果有 INF，可一并保存（可选）
        if 'INF' in visuals:
            fake_img = Metrics.tensor2img(visuals['INF'])
        else:
            fake_img = None

        # 保存图片
        base_name = f"sample_{idx:03d}"
        Metrics.save_img(hr_img, os.path.join(result_root, f'{base_name}_hr.png'))
        Metrics.save_img(sr_last, os.path.join(result_root, f'{base_name}_sr.png'))
        if fake_img is not None:
            Metrics.save_img(fake_img, os.path.join(result_root, f'{base_name}_inf.png'))

        # 计算 PSNR / SSIM（用最终 SR 和 HR）
        eval_psnr = Metrics.calculate_psnr(sr_last, hr_img)
        eval_ssim = Metrics.calculate_ssim(sr_last, hr_img)

        avg_psnr += eval_psnr
        avg_ssim += eval_ssim

        logger.info(
            f'[Image {idx}] PSNR: {eval_psnr:.4f} dB, SSIM: {eval_ssim:.4f}'
        )

    if idx > 0:
        avg_psnr /= idx
        avg_ssim /= idx

    logger.info('================ Evaluation Summary ================')
    logger.info(f'Average PSNR: {avg_psnr:.4f} dB')
    logger.info(f'Average SSIM: {avg_ssim:.4f}')
    logger.info('Results saved to: {}'.format(result_root))
    logger.info('===================================================')


if __name__ == "__main__":
    main()
