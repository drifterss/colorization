import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = self.conv(x)
        return x + residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=None, kernel_size=4, stride=2,
                 padding=1, dropout=0, activation_fn=nn.ReLU()):
        super(DownsampleBlock, self).__init__()
        model = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)]

        if normalize == 'batch':
            # + batchnorm
            model.append(nn.BatchNorm2d(out_size))
        elif normalize == 'instance':
            # + instancenorm
            model.append(nn.InstanceNorm2d(out_size))

        model.append(activation_fn)

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=None, kernel_size=4, stride=2,
                 padding=1, dropout=0.0, activation_fn=nn.ReLU()):
        super(UpsampleBlock, self).__init__()
        model = [nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)]

        if normalize == 'batch':
            # add batch norm
            model.append(nn.BatchNorm2d(out_size))
        elif normalize == 'instance':
            # add instance norm
            model.append(nn.InstanceNorm2d(out_size))

        model.append(activation_fn)

        if dropout > 0:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)

        return x


class SkipConnection(nn.Module):
    def __init__(self, channels):
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.conv(x)


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        # type(List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)  # 按通道合并
        # bn1 + relu1 + conv1
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


# growth_rate=32,
# block_config=(6, 12, 24)
# num_init_features=64
# bn_size=4
# drop_rate=0
# num_classes=256
# memory_efficient=False

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)

        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        return new_features


# 构建 3 个denseblock，num_layers 分别是 6,12,24
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))  # 尺寸减少一半


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=256, memory_efficient=False):
        super(DenseNet, self).__init__()

        # 首层卷积层
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # 构建DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):  # 构建3个DenseBlock
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)  # 添加第i个 Dense_block

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,  # 每个DenseBlock后跟一个TransitionLayer
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.fc1 = nn.Linear(1024 * 16 * 16, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        # Linear layer
        self.classifier1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.classifier2 = nn.Linear(512, num_classes)
        self.bn2 = nn.BatchNorm1d(256)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = out.view(-1, 1024 * 16 * 16)

        out = F.relu(self.bn0(self.fc1(out)))
        out = F.relu(self.bn1(self.classifier1(out)))
        out = F.relu(self.bn2(self.classifier2(out)))

        return out


class FusionNet(nn.Module):
    def __init__(self, normalization_type=None):
        super(FusionNet, self).__init__()
        self.norm = normalization_type

        self.desnet = DenseNet()

        self.down1 = DownsampleBlock(1, 64, normalize=self.norm, kernel_size=4, stride=1, padding=0, dropout=0)
        self.down2 = DownsampleBlock(64, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down3 = DownsampleBlock(64, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down4 = DownsampleBlock(128, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down5 = DownsampleBlock(256, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down6 = DownsampleBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down7 = DownsampleBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)
        self.down8 = DownsampleBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0)

        self.residual = nn.Sequential(*[ResidualBlock(512) for _ in range(8)])

        self.up1 = UpsampleBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip1 = SkipConnection(512)
        self.up2 = UpsampleBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip2 = SkipConnection(512)
        self.up3 = UpsampleBlock(512, 512, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip3 = SkipConnection(512)
        self.up4 = UpsampleBlock(512, 256, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip4 = SkipConnection(256)
        self.up5 = UpsampleBlock(256, 128, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip5 = SkipConnection(128)
        self.up6 = UpsampleBlock(128, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip6 = SkipConnection(64)
        self.up7 = UpsampleBlock(64, 64, normalize=self.norm, kernel_size=4, stride=2, padding=1, dropout=0.5)
        self.skip7 = SkipConnection(64)

    def forward(self, x):
        x_3 = torch.cat([x, x, x], dim=1)
        y1 = self.desnet(x_3)

        y2 = F.interpolate(x, size=(259, 259), mode='bilinear', align_corners=True)
        d1 = self.down1(y2)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        y1 = y1.unsqueeze(2).unsqueeze(2).expand_as(d4)
        d4 = self.skip4(y1, d4)

        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        d8 = self.residual(d8)
        u1 = self.skip1(self.up1(d8), d7)
        u2 = self.skip2(self.up2(u1), d6)
        u3 = self.skip3(self.up3(u2), d5)
        u4 = self.skip4(self.up4(u3), d4)
        u5 = self.skip5(self.up5(u4), d3)
        u6 = self.skip6(self.up6(u5), d2)
        u7 = self.skip7(self.up7(u6), d1)

        return u7
