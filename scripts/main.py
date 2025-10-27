"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.script_util_x0_enhancement import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion_RR,
    add_dict_to_argparser,
    args_to_dict,
)

from save_image_utils import save_images
from npz_dataset import NpzDataset, DummyDataset
from imagenet_dataloader.imagenet_dataset import ImageFolderDataset
from MyLoss import IBSLoss, exclusion_loss, cross_consistent_loss,PerceptualLoss,cap_loss,PhaseLoss,TVL1,lspa_loss,ColorConstancyLoss
from collections import defaultdict

import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2
import pdb
import random
import math
import torch
import torch.nn as nn
import MyLoss
from scipy.stats import entropy


import matplotlib.pyplot as plt
from time import time
import matplotlib
matplotlib.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] ='3'
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def get_dataset(path, global_rank, world_size):
    if os.path.isfile(path):  # base_samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        dataset = ImageFolderDataset(path, transform=None,
                                     permute=True, normalize=True, rank=global_rank, world_size=world_size)
    return dataset

import torch

def differentiable_entropy(img_tensor, num_bins=256, eps=1e-8):
    """
    计算 [N, C, H, W] 张量上每通道图像的可微信息熵。

    :param img_tensor: Tensor of shape [N, C, H, W], 应为 [0, 1] 范围内
    :param num_bins: 用于构建 soft histogram 的 bin 数
    :param eps: 防止 log(0)
    :return: Tensor of shape [N, C]，每通道图像的信息熵
    """
    N, C, H, W = img_tensor.shape
    img = img_tensor.reshape(N, C, -1)  # [N, C, H*W]

    # 创建 histogram bin centers
    bin_centers = torch.linspace(0.0, 1.0, steps=num_bins, device=img.device)  # [B]
    bin_centers = bin_centers.view(1, 1, 1, num_bins)  # [1, 1, 1, B]

    # 计算 soft assignment（高斯核或拉普拉斯核皆可，使用负平方差作为权重）
    img_exp = img.unsqueeze(-1)  # [N, C, H*W, 1]
    weights = torch.exp(-((img_exp - bin_centers) ** 2) / 0.01)  # 可调 sigma
    hist = weights.sum(dim=2)  # [N, C, B]

    # 归一化为概率分布
    prob = hist / (hist.sum(dim=-1, keepdim=True) + eps)  # [N, C, B]

    # 熵公式: -sum(p * log(p))
    entropy = -torch.sum(prob * torch.log(prob + eps), dim=-1)  # [N, C]
    return entropy.sum()

def l1_when_negative(x):
    """
    返回 L1 损失，当 x < 0 时为 |x|，否则为 0。
    :param x: Tensor
    :return: 标量损失
    """
    mask = (x < 0).float()          # 生成掩码
    l1_loss = torch.abs(x) * mask  # 仅对小于 0 的部分取绝对值
    return l1_loss.sum()           # 输出标量

def lin2rgb(x):
    """
    Convert linear RGB to sRGB using IEC 61966-2-1 standard gamma.
    Input x must be in [0, 1].
    """
    x = torch.clamp(x, 0.0, 1.0)
    threshold = 0.0031308
    below = x <= threshold
    above = ~below
    result = torch.zeros_like(x)
    result[below] = 12.92 * x[below]
    result[above] = 1.055 * torch.pow(x[above], 1/2.4) - 0.055
    return result
def calculate_ADoLP(img0, img45, img90, img135):
    """Calculate Angle and Degree of Linear Polarization from the four polarized images"""
    # Calculate Stokes parameters
    img0 = (img0.float().clone() + 1.0) / 2
    img45 = (img45.float().clone() + 1.0) / 2
    img90 = (img90.float().clone() + 1.0) / 2
    img135 = (img135.float().clone() + 1.0) / 2

    '''img0 = img0[:,1:2,...].repeat(1, 3, 1, 1)
    img45 = img45[:, 1:2, ...].repeat(1, 3, 1, 1)
    img90 = img90[:, 1:2, ...].repeat(1, 3, 1, 1)
    img135 = img135[:, 1:2, ...].repeat(1, 3, 1, 1)'''
    I = 0.5 * (img0 + img45 + img90 + img135)
    Q = img0 - img90
    U = img45 - img135

    # Avoid division by zero
    #Q[Q == 0] = 0.0001
    I[I == 0] = 0.000001

    # Calculate DoLP and AoLP
    DoLP = np.sqrt(Q ** 2 + U ** 2) / I
    #DoLP, _ = torch.max(DoLP, dim=1, keepdim=True)  # [B, 1, H, W]
    #DoLP = DoLP.expand(-1, 3, -1, -1)  # [B, 3, H, W]
    DoLP = DoLP.clamp(0, 1)

    '''sample_t = (DoLP * 255).clamp(0, 255).to(th.uint8)
    sample_t = sample_t.permute(0, 2, 3, 1)
    sample_t = sample_t.contiguous()
    sample_t = sample_t.detach().cpu().numpy()
    save_images(sample_t, ['dolp_0.png', ],
                "/data/cyt2/difussionRR/save_RR_2/test_linshi/")'''
    #DoLP[DoLP > 1] = 1
    AoLP = 0.5 * np.arctan2(U, Q)
    #AoLP = torch.remainder(AoLP, math.pi)/math.pi
    #AoLP = AoLP.clamp(0, 1)

    return AoLP, DoLP

def Sreflection(img, red_thresh_min=0,red_thresh_max=0.5, green_thresh=0.3, blue_thresh=0.1):
    """
    将暗红色区域设置为0，其余区域设置为1。
    输入：
        img: [B, 3, H, W]，取值范围为 [0, 1] 或 [0, 255]
    输出：
        out: 与 img 同形状的张量，值为 0 或 1
    """
    # 拆分 RGB 通道
    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]

    # 暗红色掩码：R 高，G/B 低
    dark_red_mask = (R > red_thresh_min) & (R < red_thresh_max)& (G < green_thresh) & (B < blue_thresh)  # [B, H, W]

    # 扩展到3通道
    mask_3ch = dark_red_mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [B, 3, H, W]

    # 初始化输出为全 1
    out = torch.ones_like(img)

    # 将暗红区域设为 0
    out[mask_3ch] = 0.7

    return out


def main():
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = th.device('cuda')
    save_dir = args.save_dir if len(args.save_dir) > 0 else None

    # dist_util.setup_dist()
    logger.configure(dir=save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_RR(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    '''model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )'''

    checkpoint_path = os.path.join("/data/cyt2/", "initial_weights.pt")
    torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(th.load(checkpoint_path, map_location="cpu"))
    # model.to(dist_util.dev())
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    #diffusion_steps = model_and_diffusion_defaults()["diffusion_steps"]
    ibsloss = IBSLoss(device)
    perloss = PerceptualLoss(device)
    phaseloss = PhaseLoss()
    colorloss = ColorConstancyLoss()
    # 初始化数据和图像
    '''fig, ax = plt.subplots()
    line, = ax.plot(t_values, mse_values, 'b-', label='scaled MSE')
    ax.set_xlabel('t')
    ax.set_ylabel('mse')
    ax.set_title('MSE vs t')
    #ax.set_xlim(0, diffusion_steps)  # 设置横坐标范围
    ax.set_xlim(0, args.denoise_steps)
    ax.set_ylim(0, 1)  # 设置横坐标范围
    ax.legend()
    ax.grid(False)'''
    #L_exp = MyLoss.L_exp(8, 0.3)
    #L_color = MyLoss.L_color()
    ltv = TVL1()#L_TV()
    l1_loss = nn.L1Loss()

    def light_cond_fn_all_r(x, x_reflection, t, light_factor_0=None, corner=None, x_supervised=None,
                          DoLP=None, x_input=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None, img_name=None):

        '''base_name = img_name[0].split('.')[0]
        ext = img_name[0].split('.')[1]
        new_name = f"{base_name}_{t.item()}.{ext}"
        sample = x.clone()
        sample = sample.clamp(-1, 1)
        sample = (((sample + 1) * 127.5)).clamp(0, 255).to(th.uint8)  #
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.detach().cpu().numpy()
        save_images(sample, new_name,
                    os.path.join("/data/cyt2/difussionRR/save_RR_2/", 'images_iter'))'''

        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            device_x_in_lr = x_in.device
            x_reflection_in = x_reflection.detach().requires_grad_(True)
            loss = 0
            #loss_t = 0
            loss_reflection = 0
            if not x_reflection_in is None:
                x_supervised = x_supervised[:, :, corner[0]:corner[0] + corner[2], corner[1]:corner[1] + corner[2]]
                x_input = x_input[:, :, corner[0]:corner[0] + corner[2], corner[1]:corner[1] + corner[2]]
                DoLP = DoLP[:, :, corner[0]:corner[0] + corner[2], corner[1]:corner[1] + corner[2]]

                x_in=(x_in + 1) / 2
                x_reflection_in=(x_reflection_in + 1) / 2
                #x_supervised = x_supervised.to(device_x_in_lr)
                k=5.0
                DoLP_y_r = torch.exp(k * (1 - DoLP))
                x_input = (x_input + 1) / 2
                mse_supervised_reflection = ibsloss(DoLP_y_r * x_reflection_in, DoLP_y_r * x_input)
                loss = loss - (100*mse_supervised_reflection)* args.img_guidance_scale # 100*mse_supervised_reflection
                '''print('step t %d,  mse_ref_super is %.8f' % (
                    t[0], 100*mse_supervised_reflection))# excl is %.8f, cc is %.8f,'''

                # 更新图像
                '''line.set_xdata(t_values)
                line.set_ydata(mse_values)
                ax.relim()
                ax.autoscale_view()'''

            return light_factor_0, torch.autograd.grad(loss, x_reflection_in)

    def light_cond_fn_all_t(x, x_reflection, t, light_factor_0=None, corner=None, scale=1.0, x_supervised=None,
                          Area=None, x_input=None, sample_noisy_x_lr=False, diffusion=None, sample_noisy_x_lr_t_thred=None, img_name=None):

        '''base_name = img_name[0].split('.')[0]
        ext = img_name[0].split('.')[1]
        new_name = f"{base_name}_{t.item()}.{ext}"
        sample = x.clone()
        sample = sample.clamp(-1, 1)
        sample = (((sample + 1) * 127.5)).clamp(0, 255).to(th.uint8)  #
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.detach().cpu().numpy()
        save_images(sample, new_name,
                    os.path.join("/data/cyt2/difussionRR/save_RR_2/", 'images_iter'))'''

        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            device_x_in_lr = x_in.device
            x_reflection_in = x_reflection.detach().requires_grad_(True)
            loss = 0
            #loss_t = 0
            loss_reflection = 0
            if not x_in is None:
                # x_lr and x_in are of shape BChw, BCHW, they are float type that range from -1 to 1, x_in for small t'
                x_input = x_input[:, :, corner[0]:corner[0] + corner[2], corner[1]:corner[1] + corner[2]]
                Area = Area[:, :, corner[0]:corner[0] + corner[2], corner[1]:corner[1] + corner[2]]
                #x_input = x_input.to(device_x_in_lr)

                t_coeff = torch.tanh(2 * torch.exp(5 * light_factor_0) / torch.exp(torch.tensor(5.0)))*Area #Area是针对眼镜反光
                x_in = (x_in + 1) / 2
                x_reflection_in=(x_reflection_in + 1) / 2
                x_input = (x_input + 1) / 2

                mse_0 = l1_loss(x_in, (x_input - (1 - t_coeff)/scale * x_reflection_in))  # /3 室内眼镜反光
                loss = loss - (3500*mse_0)* args.img_guidance_scale # mse_0 +mse_45+ mse_135 100*excl+100*cc

                '''print('step t %d,  mse is %.8f, cap is %.8f' % (
                    t[0], 3500*mse_0, 100*cap))# excl is %.8f, cc is %.8f,'''
                mse_values_t.append((3500 * mse_0).item())
                t_values.append(t[0])

                # 更新图像
                '''line.set_xdata(t_values)
                line.set_ydata(mse_values)
                ax.relim()
                ax.autoscale_view()'''

            return light_factor_0, torch.autograd.grad(loss, x_in)

    def model_fn(x, t):
        # assert light_factor is not None
        return model(x, t, y=None)

    logger.log("loading dataset...")
    # load gan or vae generated images
    if args.start_from_scratch and args.use_img_for_guidance:
        pass
    else:
        if args.start_from_scratch:
            dataset = DummyDataset(args.num_samples, rank=args.global_rank, world_size=args.world_size)
        else:
            dataset = get_dataset(args.dataset_path, args.global_rank, args.world_size)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # load lr images that are used for guidance
    if args.use_img_for_guidance:
        dataset_lr_0 = get_dataset(args.base_samples_0, args.global_rank, args.world_size)
        dataloader_lr_0 = th.utils.data.DataLoader(dataset_lr_0, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=16)
        dataset_l_45 = get_dataset(args.base_samples_45, args.global_rank, args.world_size)
        dataloader_l_45 = th.utils.data.DataLoader(dataset_l_45, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=16)
        dataset_lr_90 = get_dataset(args.base_samples_90, args.global_rank, args.world_size)
        dataloader_lr_90 = th.utils.data.DataLoader(dataset_lr_90, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=16)
        dataset_lr_135 = get_dataset(args.base_samples_135, args.global_rank, args.world_size)
        dataloader_lr_135 = th.utils.data.DataLoader(dataset_lr_135, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=16)
        dataset_supervised = get_dataset(args.base_samples_supervised, args.global_rank, args.world_size)
        dataloader_supervised = th.utils.data.DataLoader(dataset_supervised, batch_size=args.batch_size, shuffle=False,
                                                         num_workers=16)
        '''dataset_dolp = get_dataset(args.base_samples_dolp, args.global_rank, args.world_size)
        dataloader_dolp = th.utils.data.DataLoader(dataset_dolp, batch_size=args.batch_size, shuffle=False,
                                                         num_workers=16)'''
        if args.start_from_scratch:
            dataset = DummyDataset(len(dataset_lr_0), rank=0, world_size=1)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        dataloader = zip(dataloader, dataloader_lr_0, dataloader_l_45, dataloader_lr_90,dataloader_lr_135,
                         dataloader_supervised)

    # args.save_png_files=True
    '''if args.save_png_files:
        print(logger.get_dir())
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'images_reflection'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'mask_0'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'mask_45'), exist_ok=True)
        os.makedirs(os.path.join(logger.get_dir(), 'mask_135'), exist_ok=True)
        start_idx = args.global_rank * dataset.num_samples_per_rank'''

    logger.log("sampling...")
    all_images = []
    # while len(all_images) * args.batch_size < args.num_samples:
    t_values = []
    mse_values_t = []
    mse_values_r = []
    # 所有样本
    starttime = time()
    for i, data in enumerate(dataloader):
        # 单个样本
        '''t_values = []
        mse_values_t = []
        mse_values_r = []'''
        image, label = data[0]
        image_lr_0, img_name = data[1]
        image_lr_45, img_name = data[2]
        image_lr_90, img_name = data[3]
        image_lr_135, img_name = data[4]
        image_supervised, img_name = data[5]
        #dolp, img_name = data[6]
        #dolp = (dolp+1)/2
        aolp, dolp = calculate_ADoLP(image_lr_0, image_lr_45, image_lr_90, image_lr_135)
        image = image.to(device)
        image_supervised = image_supervised.to(device)
        dolp = dolp.to(device)
        light_factor_0 = 1-dolp#torch.exp(-k * dolp)#+th.randn(shape, device=device)/10
        ref_area=Sreflection(1-(image_supervised+1)/2)
        '''sample_t = (ref_area * 255).clamp(0, 255).to(th.uint8)  #
        sample_t = sample_t.permute(0, 2, 3, 1)
        sample_t = sample_t.contiguous()
        sample_t = sample_t.detach().cpu().numpy()
        save_images(sample_t, img_name,
                    "/data/cyt2/difussionRR/save_RR/test_linshi/")'''

        '''if args.is_indoor:
            image = image*1.5
        else:
            image = image'''

        if args.is_weak:
            scale1 = 1.5
            scale2 = 2*1.5
        else:
            scale1 = 1.0
            scale2 = 1.0

        cond_fn_all_r = lambda x, x_reflection, t, light_factor_0, corner: light_cond_fn_all_r(x, x_reflection, t,
                                                                                     light_factor_0=light_factor_0,
                                                                                     corner=corner,
                                                                                     x_supervised=image_supervised,
                                                                                     DoLP=dolp,
                                                                                     x_input = image,
                                                                                     sample_noisy_x_lr=args.sample_noisy_x_lr,
                                                                                     diffusion=diffusion,
                                                                                     sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred,
                                                                                     img_name = img_name)
        cond_fn_all_t = lambda x, x_reflection, t, light_factor_0, corner: light_cond_fn_all_t(x, x_reflection, t,
                                                           light_factor_0=light_factor_0,
                                                           corner=corner,
                                                           scale=scale2,
                                                           x_supervised=image_supervised,
                                                           Area=ref_area,
                                                           x_input=image,
                                                           sample_noisy_x_lr=args.sample_noisy_x_lr,
                                                           diffusion=diffusion,
                                                           sample_noisy_x_lr_t_thred=args.sample_noisy_x_lr_t_thred,
                                                           img_name=img_name)
        if args.start_from_scratch:
            shape = (image.shape[0], 3, 1060, 1900)
        else:
            shape = list(image.shape)

        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop_separation_2 if not args.use_ddim else diffusion.ddim_sample_loop
        ) #选取diffusion.p_sample_loop的采样方式


        sample, sample_reflection, light_factor_0 = sample_fn(
            model_fn,
            shape,
            light_factor_0,
            noise=image_supervised,
            noise_reflection=scale1*image, #*1.5 室内眼镜反光
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn_all_t=cond_fn_all_t,
            cond_fn_all_r=cond_fn_all_r,
            device=device,
            denoise_steps=args.denoise_steps,
            img_name=img_name)

        sample = sample.clamp(-1, 1)
        #sample = torch.pow((sample+1)/2.0, 1.3)
        #sample = (sample * 255).clamp(0, 255).to(th.uint8)
        sample = ((((sample + 1) * 127.5)).clamp(0, 255).to(th.uint8))  #
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.detach().cpu().numpy()

        sample_reflection = sample_reflection.clamp(-1, 1)
        #sample_reflection = torch.pow((sample_reflection+1)/2.0, 1.3)
        #sample_reflection = (sample_reflection * 255).clamp(0, 255).to(th.uint8)
        sample_reflection = ((sample_reflection + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_reflection = sample_reflection.permute(0, 2, 3, 1)
        sample_reflection = sample_reflection.contiguous()
        sample_reflection = sample_reflection.detach().cpu().numpy()

        light_factor_0 = (light_factor_0*255).clamp(0, 255).to(th.uint8)
        # light_factor_0 = (light_factor_0 * 255).clamp(0, 255).to(th.uint8)
        # light_factor_0 = light_factor_0.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        light_factor_0 = light_factor_0.permute(0, 2, 3, 1)
        light_factor_0 = light_factor_0.contiguous()
        light_factor_0 = light_factor_0.detach().cpu().numpy()


        if args.save_png_files:
            save_images(sample, img_name,
                        os.path.join(logger.get_dir(), 'images'))

            save_images(sample_reflection, img_name,
                        os.path.join(logger.get_dir(), 'images_reflection'))

            save_images(light_factor_0, img_name,
                        os.path.join(logger.get_dir(), 'mask_0'))

        all_images.append(sample)
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # dist.barrier()
    logger.log("sampling complete")
    print('Execution time = {:.0f}s'.format(time() - starttime))

    mse_sum = defaultdict(float)
    t_sum = defaultdict(float)
    for t, mse in zip(t_values, mse_values_t):
        mse_sum[int(t)] += mse

    for t in t_values:
        t_sum[int(t)] += 1

    print(dict(mse_sum))
    print(dict(t_sum))


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # add zhaoyang own's arguments
    parser.add_argument("--device", default=3, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=2, type=int, help='global rank of this process')
    parser.add_argument("--is_weak", default=False, type=bool, help='weak light source or not')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    parser.add_argument("--save_dir", default="/data/cyt2/difussionRR/test_analys_eyeglass/256/6_iter/",
                        type=str, help='the directory to save the generate images')
    '''parser.add_argument("--save_png_files", action='store_true',
                        help='whether to save the generate images into individual png files')'''
    parser.add_argument("--save_png_files", default=True, type=bool,
                        help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true',
                        help='whether to save the generate images into a single numpy array')

    # these two arguments are only valid when not start from scratch
    parser.add_argument("--denoise_steps", default=6, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path",
                        default="/data/cyt2/difussionRR/eyeglass/S0/",#_indoor
                        type=str, help='path to the generated images. Could be an npz file or an image folder')

    '''parser.add_argument("--use_img_for_guidance", action='store_true',
                        help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')'''
    parser.add_argument("--img_guidance_scale", default=5, type=float, help='guidance scale')
    parser.add_argument("--base_samples_0",
                        default='/data/cyt2/difussionRR/eyeglass/0/',
                        type=str,
                        help='the directory or npz file to the guidance imgs. This folder should have the same structure as dataset_path, there should be a one to one mapping between images in them')
    parser.add_argument("--base_samples_45",
                        default="/data/cyt2/difussionRR/eyeglass/45/",
                        type=str)
    parser.add_argument("--base_samples_90",
                        default="/data/cyt2/difussionRR/eyeglass/90/",
                        type=str)
    parser.add_argument("--base_samples_135",
                        default="/data/cyt2/difussionRR/eyeglass/135/",
                        type=str)
    parser.add_argument("--base_samples_supervised",
                        default="/data/cyt2/difussionRR/eyeglass/best/",
                        type=str)
    parser.add_argument("--base_samples_dolp",
                        default="/data/cyt2/difussionRR/eyeglass/DoLP/",
                        type=str)
    parser.add_argument("--sample_noisy_x_lr", action='store_true',
                        help='whether to first sample a noisy x_lr, then use it for guidance. ')
    parser.add_argument("--sample_noisy_x_lr_t_thred", default=1e8, type=int,
                        help='only for t lower than sample_noisy_x_lr_t_thred, we add noise to lr')
    parser.add_argument('--start_from_scratch', default=False, type=bool,help='whether to generate images purely from scratch, not use gan or vae generated samples')
    parser.add_argument('--use_img_for_guidance', default=True, type=bool,
                        help='whether to use a (low resolution) image for guidance. If true, we generate an image that is similar to the low resolution image')
    #parser.add_argument("--diffusion_steps", default=10, type=int)
    # 模型参数

    # num_samples is defined elsewhere, num_samples is only valid when start_from_scratch and not use img as guidance
    # if use img as guidance, num_samples will be set to num of guidance images
    # parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true')
    return parser


import pdb

if __name__ == "__main__":
    main()

