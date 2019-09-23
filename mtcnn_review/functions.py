# -*- coding:utf-8 -*-
import os
import cv2
import torch
import dataset
import numpy as np
from torch.nn import init


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "Linear"]:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
    elif classname.find("PReLU") != -1:
        init.constant_(m.weight.data, 0.01)


def draw(image: str, coordinate: tuple, landmark: tuple):
    img = cv2.imread(image)
    x1, y1, x2, y2 = coordinate
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for i in range(len(landmark)):
        if i % 2 != 0:
            cv2.circle(img, (landmark[i-1], landmark[i]), 3, (0, 0, 0))
    cv2.imshow("face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iou(box, boxes, ismin=False):
    """calculate the inter area between multi boxes
            box: numpy.ndarray ndim=1
            boxes: numpy.ndarray ndim=2"""
    # 计算各自面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 计算重叠面积
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    side_1 = np.maximum(0, x2 - x1)
    side_2 = np.maximum(0, y2 - y1)
    inter_area = side_1 * side_2
    # 计算IOU
    if ismin:
        IOU = inter_area / np.minimum(box_area, boxes_area)
    else:
        IOU = inter_area/(box_area + boxes_area - inter_area)
    return IOU



