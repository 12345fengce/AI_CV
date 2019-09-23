# -*-coding:utf-8-*-
import torch
import functions
import torch.nn as nn


class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super(Convolutional, self).__init__()
        self.layer = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU(),
                                    )

    def forward(self, x):
        return self.layer(x)


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv = nn.Sequential(
                                    Convolutional(3, 32, 3, 1, 0),
                                    Convolutional(32, 64, 3, 2, 1),
                                    Convolutional(64, 128, 3, 1, 0),
                                    Convolutional(128, 32, 3, 1, 0),
                                    )

        self.out_cls = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.out_offset = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, bias=False)
        self.out_landmark = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        h = self.conv(x)
        y_cls = self.out_cls(h)
        y_offset = self.out_offset(h)
        y_landmark = self.out_landmark(h)
        return y_cls, y_offset, y_landmark


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv = nn.Sequential(
                                        Convolutional(3, 32, 3, 1, 0),  # 22
                                        Convolutional(32, 64, 3, 2, 1),   # 11
                                        Convolutional(64, 128, 3, 1, 0),  # 9
                                        Convolutional(128, 256, 3, 2, 0),  # 4
                                        Convolutional(256, 128, 4, 1, 0),  # 1
                                        )
        self.out_cls = nn.Linear(in_features=128, out_features=1, bias=False)
        self.out_offset = nn.Linear(in_features=128, out_features=4, bias=False)
        self.out_landmark = nn.Linear(in_features=128, out_features=10, bias=False)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(shape=(h.size(0), -1))
        y_cls = self.out_cls(h)
        y_offset = self.out_offset(h)
        y_landmark = self.out_landmark(h)
        return y_cls, y_offset, y_landmark


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv = nn.Sequential(
                                    Convolutional(3, 32, 3, 1, 0),  # 46
                                    Convolutional(32, 64, 3, 2, 1),  # 23
                                    Convolutional(64, 128, 3, 1, 0),  # 21
                                    Convolutional(128, 256, 3, 2, 0),  # 10
                                    Convolutional(256, 512, 3, 1, 0),  # 8
                                    Convolutional(512, 1024, 3, 2, 1),  # 4
                                    Convolutional(1024, 128, 4, 1, 0),  # 1
                                    )
        self.out_cls = nn.Linear(in_features=128, out_features=1, bias=False)
        self.out_offset = nn.Linear(in_features=128, out_features=4, bias=False)
        self.out_landmark = nn.Linear(in_features=128, out_features=10, bias=False)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(shape=(h.size(0), -1))
        y_cls = self.out_cls(h)
        y_offset = self.out_offset(h)
        y_landmark = self.out_landmark(h)
        return y_cls, y_offset, y_landmark


class RLoss(nn.Module):
    """只对W参数进行惩罚，bias不进行惩罚
        model: 网络
        weight_decay: 衰减系数（加权）
        p: 范数，默认L2"""
    def __init__(self, model, weight_decay=0, p=2):
        super(RLoss, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p

    def forward(self):
        regular_loss = 0
        for param in self.model.parameters():
            if len(param.size()) > 1:
                regular_loss += torch.norm(param, self.p)
        return regular_loss*self.weight_decay




   

    


