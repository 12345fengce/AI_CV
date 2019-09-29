# -*- coding:utf-8 -*-
import os
import cv2
import torch
import dataset
import numpy as np
import torch.nn.init as init
import torchvision.transforms as tf


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
                        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])


def nms(boxes, threshold=0.3, ismin=False):
    keep = []
    if len(boxes) == 0:
        return torch.Tensor([])
    _, indices = boxes[:, 0].sort()
    boxes = boxes[indices]
    while len(boxes) != 0:
        box = boxes[-1]
        keep.append(box)
        boxes = boxes[:-1]
        if len(boxes) == 0:
            return torch.stack(keep)
        "Iou"
        _, x1, y1, x2, y2 = box
        w, h = x2-x1, y2-y1
        area = w*h
        X1, Y1, X2, Y2 = boxes[:1], boxes[:2], boxes[:3], boxes[:4]
        W, H = X2-X1, Y2-Y1
        Area = W*H
        __x1, __y1 = torch.max(x1, X1), torch.max(y1, Y1)
        __x2, __y2 = torch.min(x2, X2), torch.max(y2, Y2)
        __w, __h = __x2-__x1, __y2-__y1
        __area = __w*__h
        if ismin:
            iou = __area/torch.min(area, Area)
        else:
            iou = _area/(area+Area-__area)

        boxes = boxes[boxes[:, 0] < threshold]

    return torch.stack(keep)