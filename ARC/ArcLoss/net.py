# -*-coding:utf-8-*-
import cfg
import math
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
        self.loss = ArcLoss(in_features=cfg.FEATURE_NUM, out_features=cfg.CLS_NUM, m=cfg.m, s=cfg.s)

    def forward(self, x, labels):
        output_conv = self.conv(x)
        input_linear = torch.reshape(output_conv, shape=(output_conv.size(0), -1))
        features = self.linear(input_linear)
        wf, loss = self.loss(features, labels)
        return features, wf, loss


class ArcLoss(nn.Module):
    def __init__(self, in_features, out_features, m, s):
        """m  the size of margin
            s  the scale of features"""
        super(ArcLoss, self).__init__()
        self.m = m
        self.s = s
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x, labels, eps=1e-7):
        """x shape:  (N, in_features)"""
        for w in self.fc.parameters():
            w = F.normalize(w, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)
        wf = self.fc(x)

        # 取标签对应部分输出，然后压缩到一定范围内
        numerator = self.s * torch.cos(
            torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1+ eps, 1-eps))+self.m)
        # 取剩余部分
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        # 分母
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        # 分子/分母（化简）
        L = numerator - torch.log(denominator)
        return wf, -torch.mean(L)