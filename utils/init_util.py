import random

import numpy as np
import torch
import logging
from torch import nn, optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from model import ColorizationNet

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def init_model(args, device):
    net = ColorizationNet().to(device)

    l2_criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.0)
    # optimizer = torch.optim.Adadelta(net.parameters(),rho=0.9)

    # optimizer = getattr(optim, 'Adam')(net.parameters(), lr=args.learning_rate, weight_decay=0.0)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, patience=3, threshold=1e-4,
    #                                                     cooldown=0)

    return net, l2_criterion, optimizer


def adjust_learning_rate(optimizer, global_step, base_lr, lr_decay_rate=0.1, lr_decay_steps=6e4):
    """Adjust the learning rate of the params of an optimizer."""
    lr = base_lr * (lr_decay_rate ** (global_step / lr_decay_steps))
    if lr < 1e-6:
        lr = 1e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def log():
    logger = logging.getLogger()
    logger.setLevel('DEBUG')

    chlr = logging.StreamHandler()  # 输出到控制台的handler

    chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    fhlr = logging.FileHandler('example_new.log', encoding='utf-8', mode='a')  # 输出到文件的handler

    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger

