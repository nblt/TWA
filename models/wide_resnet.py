"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

__all__ = ['WideResNet28x10', 'WideResNet16x8']


# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


# def conv_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         init.xavier_uniform(m.weight, gain=math.sqrt(2))
#         init.constant(m.bias, 0)
#     elif classname.find('BatchNorm') != -1:
#         init.constant(m.weight, 1)
#         init.constant(m.bias, 0)


# class WideBasic(nn.Module):
#     def __init__(self, in_planes, planes, dropout_rate, stride=1):
#         super(WideBasic, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
#             )

#     def forward(self, x):
#         out = self.dropout(self.conv1(F.relu(self.bn1(x))))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out += self.shortcut(x)

#         return out


# class WideResNet(nn.Module):
#     def __init__(self, num_classes=10, depth=28, widen_factor=10, dropout_rate=0.):
#         super(WideResNet, self).__init__()
#         self.in_planes = 16

#         assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
#         n = (depth - 4) / 6
#         k = widen_factor

#         nstages = [16, 16 * k, 32 * k, 64 * k]

#         self.conv1 = conv3x3(3, nstages[0])
#         self.layer1 = self._wide_layer(WideBasic, nstages[1], n, dropout_rate, stride=1)
#         self.layer2 = self._wide_layer(WideBasic, nstages[2], n, dropout_rate, stride=2)
#         self.layer3 = self._wide_layer(WideBasic, nstages[3], n, dropout_rate, stride=2)
#         self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
#         self.linear = nn.Linear(nstages[3], num_classes)

#     def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
#         strides = [stride] + [1] * int(num_blocks - 1)
#         layers = []

#         for stride in strides:
#             layers.append(block(self.in_planes, planes, dropout_rate, stride))
#             self.in_planes = planes

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)

#         return out

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(channels)),
            ("1_activation", nn.ReLU(inplace=True)),
            ("2_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
            ("3_normalization", nn.BatchNorm2d(channels)),
            ("4_activation", nn.ReLU(inplace=True)),
            ("5_dropout", nn.Dropout(dropout, inplace=True)),
            ("6_convolution", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("0_normalization", nn.BatchNorm2d(in_channels)),
            ("1_activation", nn.ReLU(inplace=True)),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
            ("1_normalization", nn.BatchNorm2d(out_channels)),
            ("2_activation", nn.ReLU(inplace=True)),
            ("3_dropout", nn.Dropout(dropout, inplace=True)),
            ("4_convolution", nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))
        self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, num_classes: int):
        super(WideResNet, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("0_convolution", nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)),
            ("1_block", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
            ("2_block", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
            ("3_block", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
            ("4_normalization", nn.BatchNorm2d(self.filters[3])),
            ("5_activation", nn.ReLU(inplace=True)),
            ("6_pooling", nn.AvgPool2d(kernel_size=8)),
            ("7_flattening", nn.Flatten()),
            ("8_classification", nn.Linear(in_features=self.filters[3], out_features=num_classes)),
        ]))

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)

class WideResNet28x10:
    base = WideResNet
    args = list()
    kwargs = {'depth': 28, 'width_factor': 10}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

class WideResNet16x8:
    base = WideResNet
    args = list()
    kwargs = {'depth': 16, 'width_factor': 8, 'dropout': 0, 'in_channels': 3}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])