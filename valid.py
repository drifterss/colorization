
##### 测试代码
import datetime
import os
import time

import cv2
import numpy as np
import torch
from skimage.color import lab2rgb

from torch import nn
from torchvision import transforms

from Data.color_data import AverageMeter, to_rgb
from Data.dataset import create_valid_loader
from configs import get_arguments

from PIL import Image
from matplotlib import pyplot as plt



from model import ColorizationNet

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


args = get_arguments()

valid_data, valid_loader = create_valid_loader(args)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = ColorizationNet().to(device)

net.load_state_dict(torch.load('./checkpoints/200.pth'))
# net.load_state_dict(torch.load('./200.pth'))
l2_criterion = nn.MSELoss()


def test():
    test_loss = 0
    start_time = time.time()
    epoch_psnr = 0.0
    epoch_ssim = 0.0

    max_pnsr = 0.0
    max_ssim = 0.0
    net.eval()
    with torch.no_grad():

        for batch_id, (input_gray, input_ab) in enumerate(valid_loader):
            input_gray, input_ab = input_gray.to(device), input_ab.to(device)
            # print(input_gray.shape)
            print(batch_id)

            output_ab = net(input_gray).to(device)

            loss = l2_criterion(output_ab, input_ab)

            test_loss += loss.item()

            # print('***********')
            # compute pnsr
            fake_color = torch.cat([input_gray, output_ab], dim=1).squeeze(0).cpu().numpy()
            real_color = torch.cat([input_gray, input_ab], dim=1).squeeze(0).cpu().numpy()

            fake_color = fake_color.transpose((1, 2, 0))  # rescale for matplotlib
            fake_color[:, :, 0:1] = fake_color[:, :, 0:1] * 100
            fake_color[:, :, 1:3] = fake_color[:, :, 1:3] * 255 - 128
            img1 = lab2rgb(fake_color.astype(np.float64))
            # print(img1.shape)  (256, 256, 3)

            real_color = real_color.transpose((1, 2, 0))  # rescale for matplotlib
            real_color[:, :, 0:1] = real_color[:, :, 0:1] * 100
            real_color[:, :, 1:3] = real_color[:, :, 1:3] * 255 - 128
            img2 = lab2rgb(real_color.astype(np.float64))
            # print(img2.shape)

            psnr = compare_psnr(img1, img2)
            print(str(batch_id)+ '---'+ str(psnr))
            if psnr > max_pnsr:
                max_pnsr = psnr
            epoch_psnr += psnr

            ssim = compare_ssim(img1, img2, channel_axis=2)
            if ssim > max_ssim:
                max_ssim = ssim
            epoch_ssim += ssim
            # print('******************')

            # input_gray = input_gray.squeeze(0).cpu().numpy()
            input_gray = input_gray.squeeze().cpu().numpy()

            path1 = 'output_valid/gray/'
            path2 = 'output_valid/fake_color/'
            path3 = 'output_valid/real_color/'

            if not os.path.exists(path1):
                os.makedirs(path1)
            if not os.path.exists(path2):
                os.makedirs(path2)
            if not os.path.exists(path3):
                os.makedirs(path3)

            save_path = {'grayscale': path1, 'fake_colorized': path2,
                             'real_colorized': path3}


            plt.imsave(arr=img1, fname='{}{}'.format(save_path['fake_colorized'], '{}.jpg'.format(batch_id)))
            plt.imsave(arr=img2, fname='{}{}'.format(save_path['real_colorized'], '{}.jpg'.format(batch_id)))
            plt.imsave(arr=input_gray, fname='{}{}'.format(save_path['grayscale'], '{}.jpg'.format(batch_id)), cmap='gray')

            # print("Images saved successfully!")


    test_avg_loss = test_loss / (batch_id + 1)
    test_avg_psnr = epoch_psnr / len(valid_data)
    test_avg_ssim = epoch_ssim / len(valid_data)

    print("======> Val time", datetime.timedelta(seconds=int(time.time() - start_time)), end='')
    print("------- AVG Loss：{:.4f}".format(test_avg_loss))
    print("======> Avg PSNR：{:.4f}".format(test_avg_psnr))
    print("------- MAX PSNR：{:.4f}".format(max_pnsr))
    print("======> Avg SSIM：{:.4f}".format(test_avg_ssim))
    print("------- MAX SSIM：{:.4f}\n".format(max_ssim))


if __name__ == '__main__':
    test()
