# -*-coding:utf-8-*-
import cfg
import torch
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                              padding=padding, groups=groups, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(),
                                    )

    def forward(self, x):
        return self.conv(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, bottleneck, out_channels, stride, padding):
        super(BottleNeck, self).__init__()
        self.operate = nn.Sequential(
                                        Convolution(in_channels, bottleneck, 1, 1, 0, 1),
                                        Convolution(bottleneck, bottleneck, 3, stride, padding, bottleneck),
                                        nn.Conv2d(bottleneck, out_channels, 1, 1, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        )

    def forward(self, x):
        return self.operate(x)


class MobileNet(nn.Module):
    """Mobile Net V2"""
    def __init__(self, cls_num):
        super(MobileNet, self).__init__()
        self.bottleneck_2 = nn.Sequential(
                                            Convolution(3, 32, 3, 2, 1, 1),
                                            BottleNeck(32, 32, 16, 1, 1),
                                            BottleNeck(16, 96, 24, 2, 1),
                                            )
        self.bottleneck_3 = BottleNeck(24, 144, 24, 1, 1)
        self.bottleneck_4 = BottleNeck(24, 144, 32, 2, 1)
        self.bottleneck_5 = BottleNeck(32, 192, 32, 1, 1)
        self.bottleneck_6 = BottleNeck(32, 192, 32, 1, 1)
        self.bottleneck_7 = BottleNeck(32, 192, 64, 1, 1)
        self.bottleneck_8 = BottleNeck(64, 384, 64, 1, 1)
        self.bottleneck_9 = BottleNeck(64, 384, 64, 1, 1)
        self.bottleneck_10 = BottleNeck(64, 384, 64, 1, 1)
        self.bottleneck_11 = BottleNeck(64, 384, 96, 2, 1)
        self.bottleneck_12 = BottleNeck(96, 576, 96, 1, 1)
        self.bottleneck_13 = BottleNeck(96, 576, 96, 1, 1)
        self.bottleneck_14 = BottleNeck(96, 576, 160, 2, 1)
        self.bottleneck_15 = BottleNeck(160, 960, 160, 1, 1)
        self.bottleneck_16 = BottleNeck(160, 960, 160, 1, 1)
        self.output = nn.Sequential(
                                            BottleNeck(160, 960, 320, 1, 1),
                                            Convolution(320, 64, 1, 1, 0, 1),
                                            nn.Conv2d(64, 3*(5+cls_num), 1, 1),
                                                )

    def forward(self, x):
        # DownSample #1
        bottleneck_2 = self.bottleneck_2(x)
        bottleneck_3 = self.bottleneck_3(bottleneck_2)
        sum_1 = bottleneck_2+bottleneck_3
        # DownSample #2
        bottleneck_4 = self.bottleneck_4(sum_1)
        bottleneck_5 = self.bottleneck_5(bottleneck_4)
        sum_2 = bottleneck_4+bottleneck_5
        # DownSample #3
        bottleneck_6 = self.bottleneck_6(sum_2)
        sum_3 = sum_2+bottleneck_6
        bottleneck_7 = self.bottleneck_7(sum_3)
        bottleneck_8 = self.bottleneck_8(bottleneck_7)
        sum_4 = bottleneck_7+bottleneck_8
        # DownSample #4
        bottleneck_9 = self.bottleneck_9(sum_4)
        sum_5 = sum_4+bottleneck_9
        bottleneck_10 = self.bottleneck_10(sum_5)
        sum_6 = sum_5+bottleneck_10
        bottleneck_11 = self.bottleneck_11(sum_6)
        bottleneck_12 = self.bottleneck_12(bottleneck_11)
        sum_7 = bottleneck_11+bottleneck_12
        # DownSample #5
        bottleneck_13 = self.bottleneck_13(sum_7)
        sum_8 = sum_7+bottleneck_13
        bottleneck_14 = self.bottleneck_14(sum_8)
        bottleneck_15 = self.bottleneck_15(bottleneck_14)
        sum_9 = bottleneck_14+bottleneck_15

        bottleneck_16 = self.bottleneck_16(sum_9)
        sum_10 = sum_9+bottleneck_16

        return self.output(sum_10)





        
        



        






