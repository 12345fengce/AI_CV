# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                              padding=padding, groups=groups, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    # nn.Dropout(0.5),
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
    def __init__(self):
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
        self.bottleneck_17 = BottleNeck(160, 960, 320, 1, 1)
        self.feature = nn.Sequential(
                                    nn.Linear(in_features=320*7*7, out_features=2048, bias=False),
                                    nn.BatchNorm1d(2048),
                                    nn.PReLU(),
                                    )
        self.output = nn.Linear(2048, 500, bias=False)

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
        bottleneck_17 = self.bottleneck_17(sum_10)
        bottleneck_17 = torch.reshape(bottleneck_17, shape=(bottleneck_17.size(0), -1))

        features = self.feature(bottleneck_17)
        outputs = self.output(features)
        return features, outputs


class ArcSoftmax(nn.Module):
    def __init__(self, in_features, out_features, m, s, easy_margin=False):
        """m  the size of margin
            s  the scale of features"""
        super(ArcSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.easy_margin = easy_margin
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels, cls_num, eps=1e-7):
        """x shape:  (N, in_features)"""
        for w in self.fc.parameters():
            w = F.normalize(w, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)
        cosine = self.fc(x)
        sine = torch.sqrt(torch.clamp((1.0-cosine**2), eps, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        # 轻松边界
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # 复述论文
        one_hot = torch.zeros((x.size(0), 5), device="cuda").scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # positive outputs and negative
        output *= self.s
        return output


class CenterLoss(nn.Module):
    """CenterLoss convert data and labels transforms to loss
        cls_num, feature_num: int
        x: torch.Tensor  labels: torch.tensor ndim=1"""
    def __init__(self, cls_num, features_num):
        super(CenterLoss, self).__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num, features_num))

    def forward(self, x, labels):
        center = self.center[labels]  # center: ndim = 2  labels: ndim = 1  result: ndim = 2
        # bins种类  min最小值  max最大值
        count = torch.histc(labels.float(), bins=self.cls_num, min=0, max=self.cls_num - 1)[labels]
        distance = (((x-center)**2).sum(dim=-1)/count).sum()
        return distance