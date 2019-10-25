# -*- coding:utf-8 -*-
from torch import nn
from torch.nn import init
from torchvision import transforms as tf


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "Linear"]:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
    elif classname.find("PReLU") != -1:
        init.constant_(m.weight.data, 0.01)


transform = tf.Compose([
                        tf.ToTensor(),
                        tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        nn.AdaptiveMaxPool2d((224, 224)),
                        ])
