# -*- coding:utf-8 -*-
import torch
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
                                    nn.Linear(5*5*128, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 64),
                                     )

    def forward(self, x):
        output_conv = self.conv(x)
        input_linear = output_conv.reshape(shape=(output_conv.size(0), -1))
        output_linear = self.linear(input_linear)
        return output_linear


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.linear = nn.Sequential(
                                    nn.Linear(64, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 5*5*128),
                                    )
        self.convtranspose = nn.Sequential(
                                            ConvTranspose(128, 64, 3, 2, 0, 0),
                                            ConvTranspose(64, 32, 3, 1, 0, 0),
                                            ConvTranspose(32, 3, 3, 2, 1, 1),
                                            ConvTranspose(3, 1, 3, 1, 0, 0),
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
        output_encoder = self.encoder(x)
        output_decoder = self.decoder(output_encoder)
        return output_decoder


if __name__ == '__main__':
    x = torch.Tensor(10, 1, 28, 28)
    main = MainNet()
    print(main(x).shape)