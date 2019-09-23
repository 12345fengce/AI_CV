# -*- coding:utf-8 -*-
import torch
import dataset
import torch.nn as nn


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    )
    def forward(self, x):
        return self.conv(x)


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.convtranspose = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                        kernel_size=kernel_size, stride=stride,
                                                        padding=padding, output_padding=output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    )
    def forward(self, x):
        return self.convtranspose(x)


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.conv = nn.Sequential(
                                    Convolution(1, 3, 3, 1),
                                    Convolution(3, 32, 3, 2, 1),
                                    Convolution(32, 64, 3, 1),
                                    Convolution(64, 128, 3, 2),
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(5*5*128, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2),
                                     )

    def forward(self, x):
        output_conv = self.conv(x)
        input_linear = output_conv.reshape(shape=(output_conv.size(0), -1))
        output_linear = self.linear(input_linear)
        sigma, miu = output_linear[:, :1], output_linear[:, 1:]
        return sigma, miu


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.linear = nn.Sequential(
                                    nn.Linear(10, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 5*5*128),
                                    )
        self.convtranspose = nn.Sequential(
                                            ConvTranspose(128, 64, 3, 2, 0, 0),
                                            ConvTranspose(64, 32, 3, 1, 0, 0),
                                            ConvTranspose(32, 3, 3, 2, 1, 1),
                                            nn.ConvTranspose2d(3, 1, 3, 1, 0, 0),
                                            nn.Tanh(),
                                            )

    def forward(self, x):
        output_linear = self.linear(x)
        input_conv = output_linear.reshape(shape=(output_linear.size(0), -1, 5, 5))
        output_conv = self.convtranspose(input_conv)
        return output_conv


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def forward(self, x):
        sigma, miu = self.encoder(x)
        mean, var = miu, torch.exp(sigma)
        z = torch.randn((x.size(0), dataset.Z)).cuda()
        x_ = z*torch.exp(sigma)+miu
        xs = self.decoder(x_)
        return mean, var, xs


if __name__ == '__main__':
    a = torch.randn((3, 1, 28, 28))
    net = MainNet()
    mean, var, xs = net(a)
    print(mean, var, xs)
