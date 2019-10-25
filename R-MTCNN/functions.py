# -*- coding:utf-8 -*-
import torch
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

    w, h = torch.max((x2-x1), torch.tensor([0])), torch.max((y2-y1), torch.tensor([0]))
    inter_area = w*h

    if ismin:
        return inter_area/torch.min(area, areas)
    return inter_area/(area+areas-inter_area)


def nms_filter(boxes, threshold, ismin=False):
    """boxes: boxes[:, -1] == confidence"""
    index, values = torch.sort(boxes[:, -1])
    boxes = boxes[index]

    keep = []
    while len(boxes) > 1:
        box, boxes = boxes[0], boxes[1:]
        keep.append(box)
        iou = get_iou(box, boxes, ismin)
        boxes = boxes[iou < threshold]

    keep.append(boxes[0])
    return torch.stack(keep)

