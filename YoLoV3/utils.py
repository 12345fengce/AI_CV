# -*-coding:utf-8-*-
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


def __iou__(box, boxes, ismin=False):
    """calculate the inter area between multi boxes
        box: numpy.ndarray ndim=1
        boxes: numpy.ndarray ndim=2"""
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    side_1 = np.maximum(0, x2-x1)
    side_2 = np.maximum(0, y2-y1)
    inter_area = side_1*side_2
    if ismin:
        iou = inter_area/np.minimum(box_area, boxes_area)
    else:
        area = np.maximum(box_area+boxes_area-inter_area, 1)
        iou = inter_area/area
    return iou


def nms(boxes, threshold=0.3, ismin=False):
    """Non-Maximum Suppression,NMS
        boxes: numpy.ndarray ndim=2"""
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    get_box = []
    while boxes.shape[0] > 1:
        box = boxes[0]
        boxes = boxes[1:]
        get_box.append(box.tolist())
        iou = __iou__(box, boxes, ismin)
        iou = np.maximum(0.1, iou)
        boxes = boxes[np.where(iou<threshold)]
    if boxes.shape[0] > 0:
        get_box.append(boxes[0].tolist())
    return np.array(get_box)


transform = tf.Compose([tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def draw(boxes, path):
    img = cv2.imread(path)
    for cbox in boxes:
        x1, y1, x2, y2, _, cls = cbox
        color = (255*0.01*cls**2, 255*0.01*cls**2, 255*0.01*cls**2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, "{}".format(cfg.COCO_CLASS[cls]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imshow(path.split("/")[-1], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








        



