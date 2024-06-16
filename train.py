import datetime
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.color import lab2rgb
from torchvision import transforms

from Data.color_data import to_rgb
from Data.dataset import create_train_loader, create_test_loader
from configs import get_arguments
from utils.init_util import init_model, adjust_learning_rate, set_random_seed, log
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from skimage.metrics import structural_similarity as compare_ssim


logger = log()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(args, device):
    set_random_seed(100)

    train_data, train_loader = create_train_loader(args)
    test_data, test_loader = create_test_loader(args)
    # print(len(train_data))
    # print(len(test_data))

    logger.info(len(train_data))
    logger.info(len(test_data))

    net, l2_criterion, optimizer = init_model(args, device)

    model_save_dir = args.save_path

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # print('11111')
    global_step = 0

    list_loss = []

    net.train()
    for epoch in range(args.start_epoch, args.epochs):

        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print('start epoch:time is {}'.format(time_))
        logger.info('start epoch:time is {}'.format(time_))

        train_loss = 0.0

        for batch_id, (input_gray, input_ab) in enumerate(train_loader):

            input_gray, input_ab = input_gray.to(device), input_ab.to(device)
            # print(input_gray.shape)
            # print(input_ab.shape)

            # generate

            optimizer.zero_grad()
            output_ab = net(input_gray).to(device)

            l2Loss = l2_criterion(output_ab, input_ab)


            # optimizer
            l2Loss.backward()
            optimizer.step()

            train_loss += l2Loss.item()

            if batch_id % 1000 == 0:
                logger.info(
                    "======> Epoch[{}]({}/{}): Loss: {:.4f} lr:{}".format(
                        epoch, batch_id, len(train_loader),
                        l2Loss.item(),
                        optimizer.state_dict()['param_groups'][0]['lr'], ))


        # end of iter
        lr_loss = train_loss / (batch_id + 1)
        # print(lr_loss)

        time_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        logger.info("======> Epoch[{}] Complete: Avg. Loss: {:.4f} ".format(epoch, lr_loss))
        logger.info('End epoch:time is {}'.format(time_end))

        # lr_scheduler.step(lr_loss)
        global_step += 1
        adjust_learning_rate(optimizer, global_step, 0.0001, lr_decay_rate=0.1, lr_decay_steps=6e4)


        if epoch % 25 == 0:
            valid(net, l2_criterion, test_data, test_loader)
            save_model(net.state_dict(), model_save_dir, epoch)

    print('done!!!')


def save_model(net, model_save_dir, epoch):
    model_save_dir = '{}/{}.pth'.format(model_save_dir, epoch)
    torch.save(net, model_save_dir)


def valid(net, l2_criterion, test_data, test_loader):
    with torch.no_grad():
        test_loss = 0.0
        start_time = time.time()
        net.eval()
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        max_pnsr = 0.0
        max_ssim = 0.0
        for batch_id, (input_gray, input_ab) in enumerate(test_loader):
            input_gray, input_ab = input_gray.to(device), input_ab.to(device)

            # generate
            output_ab = net(input_gray)

            l2Loss = l2_criterion(output_ab, input_ab)

            test_loss += l2Loss.item()

            # compute pnsr
            fake_color = torch.cat([input_gray, output_ab], dim=1).squeeze(0).cpu().numpy()
            real_color = torch.cat([input_gray, input_ab], dim=1).squeeze(0).cpu().numpy()

            fake_color = fake_color.transpose((1, 2, 0))  # rescale for matplotlib
            fake_color[:, :, 0:1] = fake_color[:, :, 0:1] * 100
            fake_color[:, :, 1:3] = fake_color[:, :, 1:3] * 255 - 128
            img1 = lab2rgb(fake_color.astype(np.float64))
            # print(img1.shape)

            real_color = real_color.transpose((1, 2, 0))  # rescale for matplotlib
            real_color[:, :, 0:1] = real_color[:, :, 0:1] * 100
            real_color[:, :, 1:3] = real_color[:, :, 1:3] * 255 - 128
            img2 = lab2rgb(real_color.astype(np.float64))
            # print(img2.shape)

            psnr = compare_psnr(img1, img2)
            if psnr > max_pnsr:
                max_pnsr = psnr
            epoch_psnr += psnr

            ssim = compare_ssim(img1, img2,multichannel=True)
            if ssim > max_ssim:
                max_ssim = ssim
            epoch_ssim += ssim

            path1 = 'outputs/gray/'
            path2 = 'outputs/fake_color/'
            path3 = 'outputs/real_color/'
            if not os.path.exists(path1):
                os.makedirs(path1)
            if not os.path.exists(path2):
                os.makedirs(path2)
            if not os.path.exists(path3):
                os.makedirs(path3)

            for j in range(min(len(output_ab), 10)):
                save_path = {'grayscale': path1, 'fake_colorized': path2,
                             'real_colorized': path3}
                save_name = 'img-{}.jpg'.format(j)
                to_rgb(input_gray[j].cpu(), ab_input=input_ab[j].detach().cpu(), ab_output=output_ab[j].detach().cpu(),
                       save_path=save_path, save_name=save_name)

    test_avg_loss = test_loss / (batch_id+1)
    test_avg_psnr = epoch_psnr / len(test_data)
    test_avg_ssim = epoch_ssim / len(test_data)


    logger.info("======> Val time", datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info("------- AVG Loss：{:.4f}".format(test_avg_loss))
    logger.info("======> Avg PSNR：{:.4f}".format(test_avg_psnr))
    logger.info("------- MAX PSNR：{:.4f}".format(max_pnsr))
    logger.info("======> Avg SSIM：{:.4f}".format(test_avg_ssim))
    logger.info("------- MAX SSIM：{:.4f}\n".format(max_ssim))


if __name__ == '__main__':
    args = get_arguments()

    train(args, device)
