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
        self.softmax = ArcSoftmax(in_features=cfg.FEATURE_NUM, out_features=cfg.CLS_NUM, s=cfg.s, m=cfg.m)

    def forward(self, x, labels):
        output_conv = self.conv(x)
        input_linear = torch.reshape(output_conv, shape=(output_conv.size(0), -1))
        features = self.linear(input_linear)
        outputs = self.softmax(features, labels)
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

    def forward(self, x, labels, eps=1e-7):
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
        one_hot = torch.zeros((x.size(0), 10), device="cuda").scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # positive outputs and negative
        output *= self.s
        return output




