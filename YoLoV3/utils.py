# -*-coding:utf-8-*-
import os
import cv2
import math
import torch
import numpy as np


# priors
img_size = (416, 416)
prior_boxes = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
}
prior_areas = {
    13: [x*y for x, y in prior_boxes[13]],
    26: [x*y for x, y in prior_boxes[26]],
    52: [x*y for x, y in prior_boxes[52]]
}


# functions
def IOU(box, boxes, ismin=False):
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
        IOU = inter_area/np.minimum(box_area, boxes_area)
    else:
        area = np.maximum(box_area+boxes_area-inter_area, 1)
        IOU = inter_area/area
    return IOU


def NMS(boxes, threshold=0.3, ismin=False):
    """Non-Maximum Suppression,NMS
        boxes: numpy.ndarray ndim=2"""
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    get_box = []
    while boxes.shape[0] > 1:
        box = boxes[0]
        boxes = boxes[1:]
        get_box.append(box.tolist())
        iou = IOU(box, boxes, ismin)
        iou = np.maximum(0.1, iou)
        boxes = boxes[np.where(iou<threshold)]
    if boxes.shape[0] > 0:
        get_box.append(boxes[0].tolist())
    return np.array(get_box)


def one_hot(cls_max, cls_num):
    label = np.zeros((cls_max))
    label[cls_num] = 1
    return label


def load(data, size=416):
    label = {}
    data = np.array(data, dtype=np.float)
    data = np.array(np.split(data, len(data)//5))
    for key, item in prior_boxes.items():
        label[key] = np.zeros(shape=(key, key, 3, 7))  # label[key]: shape=(w, h, 3, 5+cls_num)
        for box in data:
            cls, x, y, w, h = box
            (x_offset, x_index), (y_offset, y_index) = math.modf(x*key/size), math.modf(y*key/size)
            for i, anchor in enumerate(item):
                w_offset, h_offset = np.log(w/anchor[0]), np.log(h/anchor[1])
                inter_area = min(w, anchor[0])*min(h, anchor[1])
                union_area = w*h+prior_areas[key][i]-inter_area
                iou = inter_area/union_area
                cls_num = one_hot(2, cls.astype(int)).astype(int)
                label[key][int(x_index), int(y_index), i] = np.array([iou, x_offset, y_offset, w_offset, h_offset, cls_num[0], cls_num[1]])
    return label[13], label[26], label[52]


def parse(out, size):
    index = out[..., 0] > 0
    index = index.nonzero()
    coordinates = []
    for idx in index:
        box = out[idx[0], idx[1], idx[-1]]
        iou, x_offset, y_offset, w_offset, h_offset, *cls = box
        iou = iou.item()
        x_offset, y_offset = x_offset+idx[0], y_offset+idx[1]
        x, y = x_offset/(size/416), y_offset/(size/416)
        w_anchor, h_anchor = prior_boxes[size][idx[-1]]
        w, h = w_anchor*torch.exp(w_offset), h_anchor*torch.exp(h_offset)
        x1, y1 = int(x-w/2), int(y-h/2)
        x2, y2 = x1+int(w), y1+int(h)
        cls = np.array([x.item() for x in cls])
        cls = np.where(cls > 0.5, 1, 0)
        coordinates.append([x1, y1, x2, y2, iou, cls])
    return coordinates


def draw(boxes, path):
    img = cv2.imread(path)
    for box in boxes:
        x1, y1, x2, y2, iou, cls = box
        if np.sum(cls) == 1 and np.argmax(cls) == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img, "Cat", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif np.sum(cls) == 1 and np.argmax(cls) == 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img, "Dog", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        



