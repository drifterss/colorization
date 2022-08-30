import random
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from Model.model import ColorizationNet


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

    return net, l2_criterion, optimizer


def adjust_learning_rate(optimizer, global_step, base_lr, lr_decay_rate, lr_decay_steps):
    """Adjust the learning rate of the params of an optimizer."""
    lr = base_lr * (lr_decay_rate ** (global_step / lr_decay_steps))
    if lr < 1e-6:
        lr = 1e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

