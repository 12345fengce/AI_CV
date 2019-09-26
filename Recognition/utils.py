# -*- coding:utf-8 -*-
from torch.nn import init


def weight_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "Linear"]:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
    elif classname.find("PReLU") != -1:
        init.constant_(m.weight.data, 0.01)