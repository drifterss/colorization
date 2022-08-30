from Model.basic_module import *


class ColorizationNet(nn.Module):
    """Generator architecture."""

    def __init__(self):
        super(ColorizationNet, self).__init__()

        self.funsion_net = FusionNet('instance')

        self.final = DownsampleBlock(
            64, 2, normalize=None, kernel_size=1, stride=1, padding=0, dropout=0, activation_fn=nn.Tanh()
        )

    def forward(self, x):
        x = self.funsion_net(x)

        x = self.final(x)
        return x
