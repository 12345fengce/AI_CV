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


class MyNet02(nn.Module):
    def __init__(self):
        super(MyNet02, self).__init__()
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
                                    nn.BatchNorm1d(128, momentum=0.9),
                                    nn.PReLU(),
                                    nn.Linear(128, 2, bias=False),
                                    )

    def forward(self, x):
        output_conv = self.conv(x)
        input_linear = torch.reshape(output_conv, shape=(output_conv.size(0), -1))
        outputs = self.linear(input_linear)
        return outputs


# class ArcFace(nn.Module):
#     def __init__(self, cls_num, feature_num):
#         super().__init__()
#         self.w = nn.Parameter(torch.randn(feature_num, cls_num))
#
#     def forward(self, x, s, m):
#         x_normal = F.normalize(x, p=2, dim=1)
#         w_normal = F.normalize(self.w, p=2, dim=0)
#         cosa = torch.matmul(x_normal, w_normal)/10
#         a = torch.acos(cosa)
#         new = torch.exp(torch.cos(a+m)*s*10)
#         origin = torch.sum(torch.exp(cosa*s*10), dim=1, keepdim=True)
#         softmax = new/(origin-torch.exp(cosa*s*10)+new)
#         return softmax


class ArcFace(nn.Module):
    def __init__(self, cls_num, feature_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)).cuda())
        self.func = nn.Softmax()

    def forward(self, x, s, m):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        # 将cosa变小，防止acosa梯度爆炸
        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)
        # 这里再乘回来
        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        # 这里arcsomax的概率和不为1，这会导致交叉熵损失看起来很大，且最优点损失也很大
        # print(torch.sum(arcsoftmax, dim=1))
        # exit()
        # lmsoftmax = (torch.exp(cosa) - m) / (
        #         torch.sum(torch.exp(cosa) - m, dim=1, keepdim=True) - (torch.exp(cosa) - m) + (torch.exp(cosa) - m))

        return arcsoftmax
        # return lmsoftmax
        # return self.func(torch.matmul(x_norm, w_norm))
