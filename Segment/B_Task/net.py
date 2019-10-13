# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class SameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super(SameConv, self).__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Sequential(
                                    nn.ReflectionPad2d(padding=padding),
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(),
                                    )

    def forward(self, x):
        return self.conv(x)


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
                                    SameConv(in_channels, out_channels),
                                    SameConv(out_channels, out_channels),
                                    )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, padding=0):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
                                    nn.ReflectionPad2d(padding=1),
                                    nn.Conv2d(in_channels, in_channels*2, 3, 2, padding=padding),
                                    nn.BatchNorm2d(in_channels*2),
                                    nn.PReLU(),
                                    )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        return self.up(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, cls_num):
        super(UNetPlusPlus, self).__init__()
        filters = [8, 16, 32, 64, 128]
        self.up = UpSample()
        # L0
        self.x_0_0 = Convolution(3, filters[0])
        self.x_0_1 = Convolution(filters[0]*3, filters[0])
        self.x_0_2 = Convolution(filters[0]*4, filters[0])
        self.x_0_3 = Convolution(filters[0]*5, filters[0])
        self.x_0_4 = Convolution(filters[0]*6, filters[0])
        # L1
        self.down_0_to_1 = DownSample(filters[0])
        self.x_1_0 = Convolution(filters[1], filters[1])
        self.x_1_1 = Convolution(filters[1]*3, filters[1])
        self.x_1_2 = Convolution(filters[1]*4, filters[1])
        self.x_1_3 = Convolution(filters[1]*5, filters[1])
        # L2
        self.down_1_to_2 = DownSample(filters[1])
        self.x_2_0 = Convolution(filters[2], filters[2])
        self.x_2_1 = Convolution(filters[2]*3, filters[2])
        self.x_2_2 = Convolution(filters[2]*4, filters[2])
        # L4
        self.down_2_to_3 = DownSample(filters[2])
        self.x_3_0 = Convolution(filters[3], filters[3])
        self.x_3_1 = Convolution(filters[3]*3, filters[3])
        # L5
        self.down_3_to_4 = DownSample(filters[3])
        self.x_4_0 = Convolution(filters[4], filters[4])
        # Output
        self.out = nn.Sequential(
                                    nn.Conv2d(in_channels=filters[0], out_channels=cls_num, kernel_size=1, stride=1),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        # x_*_0
        x_0_0 = self.x_0_0(x)
        x_1_0 = self.x_1_0(self.down_0_to_1(x_0_0))
        x_2_0 = self.x_2_0(self.down_1_to_2(x_1_0))
        x_3_0 = self.x_3_0(self.down_2_to_3(x_2_0))
        x_4_0 = self.x_4_0(self.down_3_to_4(x_3_0))
        # x_*_1
        x_0_1 = self.x_0_1(torch.cat((x_0_0, self.up(x_1_0)), dim=1))
        x_1_1 = self.x_1_1(torch.cat((x_1_0, self.up(x_2_0)), dim=1))
        x_2_1 = self.x_2_1(torch.cat((x_2_0, self.up(x_3_0)), dim=1))
        x_3_1 = self.x_3_1(torch.cat((x_3_0, self.up(x_4_0)), dim=1))
        # x_*_2
        x_0_2 = self.x_0_2(torch.cat((x_0_0, x_0_1, self.up(x_1_1)), dim=1))
        x_1_2 = self.x_1_2(torch.cat((x_1_0, x_1_1, self.up(x_2_1)), dim=1))
        x_2_2 = self.x_2_2(torch.cat((x_2_0, x_2_1, self.up(x_3_1)), dim=1))
        # x_*_3
        x_0_3 = self.x_0_3(torch.cat((x_0_0, x_0_1, x_0_2, self.up(x_1_2)), dim=1))
        x_1_3 = self.x_1_3(torch.cat((x_1_0, x_1_1, x_1_2, self.up(x_2_2)), dim=1))
        # x_*_4
        x_0_4 = self.x_0_4(torch.cat((x_0_0, x_0_1, x_0_2, x_0_3, self.up(x_1_3)), dim=1))
        return self.out(x_0_4)