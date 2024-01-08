import cv2
import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import scipy.io
from option import opt
import paths
from skimage.metrics import structural_similarity as compare_ssim
import torchvision.transforms as transforms


def shift(inputs, bias_list):
    shift_mask = torch.Tensor(
        np.zeros((inputs.shape[0], opt.channel, opt.time_ratio, opt.size, opt.size + bias_list[-1])))
    for i in range(opt.channel):
        shift_mask[:, i, :, :, bias_list[i]:bias_list[i] + opt.size] = inputs[:, :, :, :]
    return shift_mask.cuda()


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    pixel_max = 1
    return 20 * math.log10(pixel_max / math.sqrt(mse))


def PSNR_valid(result, gt):
    batch_size = gt.shape[0]
    psnr_list = []
    for b in range(batch_size):
        p = 0
        for c in range(opt.channel):
            for t in range(opt.time_ratio):
                p += psnr(result[b, c, t, :, :].cpu().numpy() * 255,
                          gt[b, c, t, :, :].cpu().numpy() * 255)
        psnr_list.append(p / (opt.channel * opt.time_ratio))
    return psnr_list


def SSIM_valid(result, gt):
    batch_size = gt.shape[0]
    ssim_list = []
    for b in range(batch_size):
        p = 0
        for c in range(opt.channel):
            for t in range(opt.time_ratio):
                p += compare_ssim(result[b, c, t, :, :].cpu().numpy() * 255,
                                  gt[b, c, t, :, :].cpu().numpy() * 255,
                                  data_range=255.0)
        ssim_list.append(p / (opt.channel * opt.time_ratio))
    return ssim_list


def prepare_path(path):
    file_paths = glob.glob(path + '*.mat')
    num_files = len(file_paths)
    return file_paths, num_files


def init_input(input, bias_list):
    result = torch.unsqueeze(input[:, :, :opt.size], dim=1)
    for bias in bias_list[1:]:
        result = torch.cat((result, torch.unsqueeze(input[:, :, bias:bias + opt.size], dim=1)), dim=1)
    return result


def shift_sum(input, bias_list):
    result = torch.Tensor(np.zeros((input.shape[0], opt.size, opt.size + bias_list[-1]))).cuda()
    for b in range(input.shape[0]):
        for c in range(opt.channel):
            result[b, :, bias_list[c]:bias_list[c] + opt.size] += input[b, c, :, :]
    return result


def train_show(result, gt, label, epoch, writer):
    for c in range(opt.channel):
        img = torch.cat((result[0, c, 0, :, :].cpu(),
                         torch.from_numpy(
                             np.zeros([opt.size, 5], dtype=np.float32)),
                         gt[0, c, 0, :, :].cpu()),
                        dim=1)
        for t in range(1, opt.time_ratio):
            img_time = torch.cat((result[0, c, t, :, :].cpu(),
                                  torch.from_numpy(
                                      np.zeros([opt.size, 5], dtype=np.float32)),
                                  gt[0, c, t, :, :].cpu()),
                                 dim=1)
            img = torch.cat(
                (img, torch.from_numpy(np.zeros([5, img_time.shape[1]], dtype=np.float32)), img_time), dim=0)
        label_name = f"{label}_{c}"
        writer.add_image(label_name, img, global_step=epoch, dataformats='HW')
    return


def train_show_gan(result, gt, label, epoch, writer):
    for c in range(opt.channel):
        img = torch.cat((result[0, c, 0, :, :].cpu(),
                         torch.from_numpy(
                             np.zeros([opt.size, 5], dtype=np.float32)),
                         gt[0, c, :, :].cpu()),
                        dim=1)
        for t in range(1, opt.time_ratio):
            img_time = torch.cat((result[0, c, t, :, :].cpu(),
                                  torch.from_numpy(
                                      np.zeros([opt.size, 5], dtype=np.float32)),
                                  gt[0, c, t, :, :].cpu()),
                                 dim=1)
            img = torch.cat(
                (img, torch.from_numpy(np.zeros([5, img_time.shape[1]], dtype=np.float32)), img_time), dim=0)
        label_name = f"{label}_{c}"
        writer.add_image(label_name, img, global_step=epoch, dataformats='HW')
    return


def test_save(mea, result, gt, psnr, ssim, index):
    save_path = os.path.join(paths.result_path, f"scene_{index}") + '/'
    txt_path = os.path.join(save_path, 'zlog.txt')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for c in range(opt.channel):
        for t in range(opt.time_ratio):
            cv2.imwrite(os.path.join(save_path, f"channel_{c}_frame_{t}.tif"),
                        (result[0, c, t, :, :].cpu().numpy() * 65535.0).astype(np.uint16))
            cv2.imwrite(os.path.join(save_path, f"gt_channel_{c}_frame_{t}.tif"),
                        (gt[0, c, t, :, :].cpu().numpy() * 65535.0).astype(np.uint16))
            pass
    cv2.imwrite(os.path.join(save_path, f"mea.tif"), (mea[0, :, :].cpu().numpy() * 65535.0).astype(np.uint16))
    with open(txt_path, 'a') as file:
        file.write(
            f"index:{index}, PSNR:{psnr[0]:.4f}, SSIM:{ssim[0]:.4f}\n")
    return


def log_save_sim(psnr, ssim):
    save_path = os.path.join(paths.result_path, 'sim') + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log_all.txt')
    with open(txt_path, 'a') as file:
        file.write(
            f"PSNR:{psnr:.4f}, SSIM:{ssim:.4f}\n")
    return
