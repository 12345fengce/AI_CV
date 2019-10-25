# -*- coding:utf-8 -*-
import os
import cv2
import cfg
import math
import torch
import numpy as np
import PIL.Image as Image
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


def get_iou(box, boxes, ismin=False):
    """box: shape(4)
        boxes: shape(n, 4)"""
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("box must be a rectangle!")
    if all(boxes[:, 0] >= boxes[:, 2]) or all(boxes[:, 1] >= boxes[:, 3]):
        raise ValueError("boxes must be the set of rectangle!")
    area = (box[2]-box[0])*(box[3]-box[1])
    areas = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])

    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    w, h = torch.max((x2-x1), torch.Tensor([0])), torch.max((y2-y1), torch.Tensor([0]))
    inter_area = w*h

    if ismin:
        return inter_area/torch.min(area, areas)
    return inter_area/(area+areas-inter_area)


def nms_filter(boxes, threshold, ismin=False):
    """boxes: boxes[:, -1] == confidence"""
    values, index = torch.sort(boxes[:, -1])
    boxes = boxes[index]

    keep = []
    while len(boxes) > 1:
        box, boxes = boxes[0], boxes[1:]
        keep.append(box)
        iou = get_iou(box, boxes, ismin)
        boxes = boxes[iou < threshold]

    if len(boxes) != 0:
        keep.append(boxes[0])
    return torch.stack(keep)


transform = tf.Compose([
                        tf.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                        tf.ToTensor(),
                        tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])


def draw(boxes, path):
    img = cv2.imread(path)
    h, w, _ = img.shape
    h_scale, w_scale = h/cfg.IMG_SIZE, w/cfg.IMG_SIZE

    for cbox in boxes:
        x1, y1, x2, y2, cls, _ = cbox
        cls = int(cls)
        x1, x2 = int(x1*w_scale), int(x2*w_scale)
        y1, y2 = int(y1*h_scale), int(y2*h_scale)
        color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, "{}".format(cfg.NAME[cls]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow(path.split("/")[-1], img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()








        



