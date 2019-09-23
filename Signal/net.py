# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
                                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.InstanceNorm1d(out_channels),
                                    nn.ReLU(),
                                    )
    def forward(self, x):
        return self.conv(x)


class SignNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
                                    Convolution(1, 16, 3, 1),
                                    Convolution(16, 32, 3, 1),
                                    Convolution(32, 64, 3, 1),
                                    Convolution(64, 128, 3, 1),
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(128, 1),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        x = x.reshape(shape=(1, 1, -1))
        output_conv = self.conv(x)
        input_linear = output_conv.reshape(shape=(1, -1))
        output_linear = self.linear(input_linear)
        return output_linear


if __name__ == '__main__':
    net = SignNet()
    x = [i for i in range(9)]
    print(net(x))