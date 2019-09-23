# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# Net for MNIST
class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Convolution, self).__init__()
        self.Convolution = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                                        nn.BatchNorm2d(out_channels),
                                        nn.LeakyReLU(0.1),
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
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(32*6*6, 128),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(128, 2, bias=False),
                                    )
        self.classify = ArcFace(feature_num=2, cls_num=10)

    def forward(self, x):
        output_conv = self.conv(x)
        input_linear = torch.reshape(output_conv, shape=(output_conv.size(0), -1))
        features = self.linear(input_linear)
        outputs = self.classify(features)
        return features, outputs


class ArcFace(nn.Module):
    def __init__(self, feature_num, cls_num):
        super(ArcFace, self).__init__()
        self.W = nn.Parameter(torch.randn((feature_num, cls_num)))

    def forward(self, X, m=0.5):
        w = F.normalize(self.W, dim=0)
        x = torch.norm(X, dim=1)
        cosa = torch.matmul(X, w)
        a = torch.acos(cosa)
        print(torch.cos(a+m))
        element = torch.exp(x*torch.cos(a+m))
        deno = torch.sum(torch.exp(cosa), dim=1)-torch.exp(cosa)+element
        return element/deno


if __name__ == '__main__':
    loss = ArcFace(2, 10)
    X = torch.arange(20).reshape((10, 2)).float()/20
    output = loss(X)
    print("output >>> ", torch.sum(output, dim=1))