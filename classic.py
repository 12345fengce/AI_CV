# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


# Shuffle Net
class ShuffleNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(ShuffleNet, self).__init__()
        self.groups = groups
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels or out_channels must be divisible by groups")
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                              padding=padding, groups=groups, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    )

    def forward(self, x):
        y = self.conv(x)
        y = y.reshape(shape=(y.size(0), int(y.size(1)/self.groups), self.groups, y.size(2), y.size(3)))
        y = y.transpose(2, 1)
        y = y.reshape(shape=(y.size(0), -1, y.size(3), y.size(4)))
        return y


# DW Net
class DWNet(nn.Module):
    def __init__(self):
        super(DWNet, self).__init__()
