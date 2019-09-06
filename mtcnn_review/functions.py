# -*- coding:utf-8 -*-
import os
import cv2
import torch
import dataset
import numpy as np


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


def iou(box, boxes):
    if len(box) == 0 or len(boxes) == 0:
        return None
    x = np.minimum(box[2], boxes[:, 2])-np.maximum(box[0], boxes[:, 0])
    y = np.minimum(box[3], boxes[:, 3])-np.maximum(box[1], boxes[:, 1])
    inter_area = x*y
    union_area = (box[3]-box[1])*(box[2]-box[0])+(boxes[:, 3]-boxes[:, 1])*(boxes[:, 2]-boxes[:, 0])-inter_area
    return inter_area/union_area




