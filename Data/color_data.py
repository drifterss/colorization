import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from torch import nn
from torchvision import datasets, transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class GrayscaleImageFolder(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''

    # root: 在指定的root路径下面寻找图片
    # transform: 对PIL
    # Image进行转换操作, transform
    # 输入是loader读取图片返回的对象
    # target_transform: 对label进行变换
    # loader: 指定加载图片的函数，默认操作是读取PILimage对象

    def __getitem__(self, index):
        # self.imgs: 返回所有图片的路径和对应的 label
        path, target = self.imgs[index]

        img = self.loader(path)  # 读取的图像类型是 PIL
        if self.transform is not None:
            img_original = self.transform(img)
            # 将参数转换成数组
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            # 转换为 tensor
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = rgb2gray(img_original)
            # 转换为 tensor
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()

        return img_original, img_ab


class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_rgb(grayscale_input, ab_input, ab_output, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
       Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf()  # clear matplotlib
    fake_color = torch.cat((grayscale_input, ab_output), 0).numpy()  # combine channels
    fake_color = fake_color.transpose((1, 2, 0))  # rescale for matplotlib
    fake_color[:, :, 0:1] = fake_color[:, :, 0:1] * 100
    fake_color[:, :, 1:3] = fake_color[:, :, 1:3] * 255 - 128
    fake_color = lab2rgb(fake_color.astype(np.float64))

    real_color = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
    real_color = real_color.transpose((1, 2, 0))  # rescale for matplotlib
    real_color[:, :, 0:1] = real_color[:, :, 0:1] * 100
    real_color[:, :, 1:3] = real_color[:, :, 1:3] * 255 - 128
    real_color = lab2rgb(real_color.astype(np.float64))

    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=fake_color, fname='{}{}'.format(save_path['fake_colorized'], save_name))
        plt.imsave(arr=real_color, fname='{}{}'.format(save_path['real_colorized'], save_name))
