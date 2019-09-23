# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


# Conv_Model
class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super(Convolution, self).__init__()
        self.layer = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.layer(x)


class ConvTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, outpadding, bias=False):
        super(ConvTrans, self).__init__()
        self.layer = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, outpadding, bias=bias),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(0.2, True))

    def forward(self, x):
        return self.layer(x)


# Generator
class GenNet(nn.Module):
    def __init__(self):
        super(GenNet, self).__init__()
        self.generator = nn.Sequential(
                                        ConvTrans(512, 256, 4, 1, 0, 0),  # 4
                                        ConvTrans(256, 128, 4, 2, 1, 0),  # 8
                                        ConvTrans(128, 64, 4, 2, 1, 0),  # 16
                                        ConvTrans(64, 32, 4, 2, 1, 0),  # 32
                                        nn.ConvTranspose2d(32, 3, 5, 3, 1, 0),  # 96
                                        nn.Tanh(),
                                        )

    def forward(self, x):
        return self.generator(x)


# Arbiter
class ArbiNet(nn.Module):
    def __init__(self):
        super(ArbiNet, self).__init__()
        self.arbiter = nn.Sequential(
                                        nn.Conv2d(3, 32, 5, 3, 1),  # 32
                                        nn.LeakyReLU(0.2, True),
                                        Convolution(32, 64, 4, 2, 1),  # 16
                                        Convolution(64, 128, 4, 2, 1),  # 8
                                        Convolution(128, 256, 4, 2, 1),  # 4
                                        nn.Conv2d(256, 10, 4, 1, 0),  # 1
                                        nn.Sigmoid(),
                                    )

    def forward(self, x):
        return self.arbiter(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "ConvTranspose2d"]:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    generator = GenNet()
    arbiter = ArbiNet()

    # generator.apply(weights_init)
    # arbiter.apply(weights_init)

    fx = torch.randn((10, 512, 1, 1))
    gen = generator(fx)
    print("gen >>> ", gen.shape)

    cls = arbiter(gen)
    print("ari >>> ", cls.shape)

