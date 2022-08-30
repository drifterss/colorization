import datetime
import os
import time

import torch

from Data.color_data import to_rgb
from Data.dataset import create_train_loader, create_test_loader
from configs import get_arguments
from utils.init_util import init_model, adjust_learning_rate, set_random_seed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(args, device):
    set_random_seed(100)

    train_data, train_loader = create_train_loader(args)
    test_data, test_loader = create_test_loader(args)

    net, l2_criterion, optimizer = init_model(args, device)

    model_save_dir = args.save_path

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    global_step = 0

    net.train()
    for epoch in range(args.start_epoch, args.epochs):

        time_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('start epoch:time is {}'.format(time_))

        train_loss = 0.0

        for batch_id, (input_gray, input_ab) in enumerate(train_loader):

            input_gray, input_ab = input_gray.to(device), input_ab.to(device)

            optimizer.zero_grad()
            output_ab = net(input_gray).to(device)

            l2Loss = l2_criterion(output_ab, input_ab)
            # optimizer
            l2Loss.backward()
            optimizer.step()

            train_loss += l2Loss.item()

            if batch_id % 10 == 0:
                print(
                    "======> Epoch[{}]({}/{}): Loss: {:.4f} lr:{}".format(
                        epoch, batch_id, len(train_loader),
                        l2Loss.item(),
                        optimizer.state_dict()['param_groups'][0]['lr'], ))

        # end of iter
        lr_loss = train_loss / (batch_id + 1)

        time_end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("======> Epoch[{}] Complete: Avg. Loss: {:.4f} ".format(epoch, lr_loss))
        print('End epoch:time is {}'.format(time_end))

        # lr
        global_step += 1
        adjust_learning_rate(optimizer, global_step, 0.0001, lr_decay_rate=args.lr_decay_rate,
                             lr_decay_steps=args.lr_decay_steps)

        if epoch % 10 == 0:
            valid(net, l2_criterion, test_loader)
            save_model(net.state_dict(), model_save_dir, epoch)

    print('done!!!')


def save_model(net, model_save_dir, epoch):
    model_save_dir = '{}/{}.pth'.format(model_save_dir, epoch)
    torch.save(net, model_save_dir)


def valid(net, l2_criterion, test_loader):
    with torch.no_grad():
        test_loss = 0.0
        start_time = time.time()
        net.eval()

        for batch_id, (input_gray, input_ab) in enumerate(test_loader):
            input_gray, input_ab = input_gray.to(device), input_ab.to(device)

            output_ab = net(input_gray)

            l2Loss = l2_criterion(output_ab, input_ab)
            test_loss += l2Loss.item()

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

    test_avg_loss = test_loss / (batch_id + 1)

    print("======> Val time", datetime.timedelta(seconds=int(time.time() - start_time)), end='')
    print("------- AVG Loss：{:.4f}".format(test_avg_loss))


if __name__ == '__main__':
    args = get_arguments()

    train(args, device)
