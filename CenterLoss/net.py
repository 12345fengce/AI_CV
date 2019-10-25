# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Convolution, self).__init__()
        self.Convolution = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False),
                                        nn.BatchNorm2d(out_channels),
                                        nn.PReLU(),
                                        )

    def forward(self, data):
        return self.Convolution(data)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Sequential(
                                    Convolution(1, 32, 1, 1),
                                    Convolution(32, 64, 3, 2),
                                    Convolution(64, 32, 1, 1),
                                    Convolution(32, 64, 3, 2),
                                    Convolution(64, 32, 1, 1),
                                    Convolution(32, 64, 3, 1),
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(64*4*4, 128, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.PReLU(),
                                    nn.Linear(128, 2),
                                    )
        self.classify = nn.Linear(in_features=2, out_features=10, bias=False)

    def forward(self, x):
        output_conv = self.conv(x)
        input_linear = torch.reshape(output_conv, shape=(output_conv.size(0), -1))
        features = self.linear(input_linear)
        outputs = self.classify(features)
        return features, outputs


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