import torch
from torch import nn

import torch.nn.functional as F

from Model.basic_module import *


class Down_net(nn.Module):
    """Generator architecture."""

    def __init__(self, normalization_type=None):
        super(Down_net, self).__init__()
        self.norm = normalization_type
        self.inConv1 = nn.Sequential(ResidualBlock(1))

    def forward(self, x):
        out = self.inConv1(x)


        return out


class ColorizationNet(nn.Module):
    """Generator architecture."""

    def __init__(self):
        super(ColorizationNet, self).__init__()
        # self.down_net = Down_net()
        self.funsion_net = FusionNet('instance')
        # self.up_net = Up_net()

        self.final = DownsampleBlock(
            64, 2, normalize=None, kernel_size=1, stride=1, padding=0, dropout=0, activation_fn=nn.Tanh()
        )

    def forward(self, x):
        # x = self.down_net(x)

        x = self.funsion_net(x)

        x = self.final(x)
        return x


if __name__ == '__main__':
    x = torch.randn([16, 1, 256, 256])
    # y = torch.randn([16, 64, 256, 256])
    #
    # net = Down_net()
    #
    # x = net(x)
    # print(x.shape)

    color_net = ColorizationNet()
    x = color_net(x)
    print(x.shape)
