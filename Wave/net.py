# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
                                    nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False),
                                    nn.BatchNorm1d(out_channels),
                                    nn.Dropout(0.5),
                                    nn.PReLU(),
                                    )

    def forward(self, x):
        return self.conv(x)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.init = nn.InstanceNorm1d(2)
        self.conv = nn.Sequential(
                                    Convolution(2, 16, 8, 4),  # 11024
                                    Convolution(16, 32, 8, 4),  # 2755
                                    Convolution(32, 64, 8, 4),  # 687
                                    Convolution(64, 128, 8, 4),  # 170
                                    Convolution(128, 32, 1, 1),  # 170
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(in_features=170*32, out_features=1024, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.Dropout(0.8),
                                    nn.PReLU(),
                                    nn.Linear(in_features=1024, out_features=128, bias=False),
                                    nn.BatchNorm1d(128),
                                    nn.Dropout(0.8),
                                    nn.PReLU(),
                                    nn.Linear(in_features=128, out_features=10),
                                    )

    def forward(self, x):
        x = self.init(x)
        y = self.conv(x)
        y = y.reshape((x.size(0), -1))
        return self.linear(y)


