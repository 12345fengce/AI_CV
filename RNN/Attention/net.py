# -*-coding:utf-8-*-
import torch
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(0.1),
                                    )

    def forward(self, x):
        return self.conv(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(
                                    Convolution(in_channels, out_channels, 1, 1, 0),
                                    Convolution(out_channels, out_channels, 3, 1, 1),
                                    Convolution(out_channels, in_channels, 1, 1, 0),
                                    )

    def forward(self, x):
        return self.module(x)+x


class HighLight(nn.Module):
    def __init__(self):
        super(HighLight, self).__init__()
        self.branch = nn.Sequential(
                                    Convolution(3, 32, 1, 1, 0),
                                    Convolution(32, 32, 3, 1, 1),
                                    nn.Conv2d(32, 3, 1, 1),
                                    )
        self.trunk = nn.Sequential(
                                    Convolution(3, 32, 3, 1, 1),
                                    Convolution(32, 64, 3, 2, 1),
                                    Residual(64, 32),
                                    Convolution(64, 128, 3, 2, 1),
                                    Residual(128, 64), Residual(128, 64),
                                    Convolution(128, 256, 3, 2, 1),
                                    Residual(256, 128), Residual(256, 128), Residual(256, 128), Residual(256, 128),
                                    )
        self.pool = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(6),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.1),
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(in_features=6*6*256, out_features=1),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        output_branch = self.branch(x)
        mask = torch.sigmoid(output_branch)
        attention = input_trunk = mask*x
        output_trunk = self.trunk(input_trunk)
        output_pool = self.pool(output_trunk)
        input_linear = output_pool.reshape(shape=(output_pool.size(0), -1))
        confi = self.linear(input_linear)
        return attention, confi


if __name__ == '__main__':
    net = HighLight()
    x = torch.Tensor(1, 3, 48, 48)
    attention, confi = net(x)
    print(attention.size(), "--attention")
    print(confi, "--confi")

