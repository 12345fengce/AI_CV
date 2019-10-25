# -*-coding:utf-8-*-
import cfg
import torch
import torch.nn as nn


class Convolutional(nn.Module):
    """Convolutional Module:
        Conv2d Layer
        BN Layer
        Activate Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super(Convolutional, self).__init__()
        self.layer = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU())

    def forward(self, x):
        return self.layer(x)


class ConvLayer(nn.Module):
    """ConvLayer Module(std):
        Convolutional 1*1 channels//2
        Convolutional 3*3 channels*2
        Add Residual"""
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.pointwise = Convolutional(in_channels, out_channels, 1, 1)
        self.convolution = Convolutional(out_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        out_point = self.pointwise(x)
        out_conv = self.convolution(out_point)
        return x+out_conv
        

class ConvSet(nn.Module):
    """ConvSet Module:
        Convolutional 1*1 channels//2
        Convolutional 3*3 channels*2
        Convolutional 1*1 channels//2
        Convolutional 3*3 channels*2
        Convolutional 1*1 channels//2"""
    def __init__(self, in_channels, out_channels):
        super(ConvSet, self).__init__()
        self.set = nn.Sequential(
                                    Convolutional(in_channels, out_channels, 1, 1),
                                    Convolutional(out_channels, in_channels, 3, 1, 1),
                                    Convolutional(in_channels, out_channels, 1, 1),
                                    Convolutional(out_channels, in_channels, 3, 1, 1),
                                    Convolutional(in_channels, out_channels, 1, 1))

    def forward(self, x):
        return self.set(x)


class UpSample(nn.Module):
    """UpSample Module:
        Convolutional 1*1 channels//2
        UpSample Layer w*2 h*2"""
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.convolution = Convolutional(in_channels, out_channels, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        out_conv = self.convolution(x)
        out_up = self.upsample(out_conv)
        return out_up


class YoLoNet(nn.Module):
    """Main Net:
        Convolutional  3 1 1
        Convolutional  3 2 1
        ConvLayer  *1
        Convolutional  3 2 1
        ConvLayer  *2
        Convolutional  3 2 1
        ConvLayer  *8
        Convolutional  3 2 1
        ConvLayer  *8
        Convolutional  3 2 1
        ConvLayer  *4
        ConvSet→Convolutional 3 1 1→Conv2d 1 1 1
        UpSample→Route Residual
        ConvSet→Convolutional 3 1 1→Conv2d 1 1 1
        UpSample→Route Residual
        ConvSet→Convolutional 3 1 1→Conv2d 1 1 1"""
    def __init__(self):
        super(YoLoNet, self).__init__()
        self.darknet1 = nn.Sequential(
                                        Convolutional(3, 32, 3, 1, 1),
                                        Convolutional(32, 64, 3, 2, 1),
                                        ConvLayer(64, 32),
                                        Convolutional(64, 128, 3, 2, 1),
                                        ConvLayer(128, 64), ConvLayer(128, 64),
                                        Convolutional(128, 256, 3, 2, 1),
                                        ConvLayer(256, 128), ConvLayer(256, 128), ConvLayer(256, 128), ConvLayer(256, 128), ConvLayer(256, 128), ConvLayer(256, 128), ConvLayer(256, 128), ConvLayer(256, 128)
                                        )
        self.darknet2 = nn.Sequential(
                                        Convolutional(256, 512, 3, 2, 1),
                                        ConvLayer(512, 256), ConvLayer(512, 256), ConvLayer(512, 256), ConvLayer(512, 256), ConvLayer(512, 256), ConvLayer(512, 256), ConvLayer(512, 256), ConvLayer(512, 256)
                                        )
        self.darknet3 = nn.Sequential(
                                        Convolutional(512, 1024, 3, 2, 1),
                                        ConvLayer(1024, 512), ConvLayer(1024, 512), ConvLayer(1024, 512), ConvLayer(1024, 512)
                                        )
        self.set1 = ConvSet(1024, 512)
        self.out1 = nn.Sequential(
                                    Convolutional(512, 1024, 3, 1, 1),
                                    nn.Conv2d(1024, 3*(5+len(cfg.COCO_CLASS)), 1, 1)
                                    )  # Out 1
        self.up1 = UpSample(512, 256)
        self.set2 = ConvSet(768, 384)  # UpSample created double channels
        self.out2 = nn.Sequential(
                                    Convolutional(384, 768, 3, 1, 1),
                                    nn.Conv2d(768, 3*(5+len(cfg.COCO_CLASS)), 1, 1)
                                    )  # Out 2
        self.up2 = UpSample(384, 192)  # UpSample created double channels
        self.out3 = nn.Sequential(
                                    ConvSet(448, 224),
                                    Convolutional(224, 448, 3, 1, 1),
                                    nn.Conv2d(448, 3*(5+len(cfg.COCO_CLASS)), 1, 1)
                                    )  # Out 3

    def forward(self, x):
        dn_1 = self.darknet1(x)  # size=52*52 To torch.cat()

        dn_2 = self.darknet2(dn_1)  # size=26*26 To torch.cat()
        dn_3 = self.darknet3(dn_2)  # size=13*13 To torch.cat()

        s_1 = self.set1(dn_3)
        o_1 = self.out1(s_1)  # ConvSet→Out 1  

        up_1 = self.up1(s_1)
        c_1 = torch.cat((up_1, dn_2), dim=1)  # UpSample create double channels

        s_2 = self.set2(c_1)
        o_2 = self.out2(s_2)  # ConvSet→Out 2

        up_2 = self.up2(s_2)
        c_2 = torch.cat((up_2, dn_1), dim=1)  # UpSample create double channels
        
        o_3 = self.out3(c_2)  # ConvSet→Out 3
        return o_1, o_2, o_3




        
        



        






