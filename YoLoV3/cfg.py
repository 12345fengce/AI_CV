# -*- coding:utf-8 -*-
"Global Variable"
BATCH = 3

EPOCH = 33333

IMG_SIZE = 416

THRESHOLD = 0.5

PATH = "F:/YOLO_V3/train"

ANCHOR_BOX = {
    13: [[116, 90], [156, 198], [373, 326]],
    26: [[30, 61], [62, 45], [59, 119]],
    52: [[10, 13], [16, 30], [33, 23]]
                }

ANCHOR_AREA = {
    13: [x*y for x, y in ANCHOR_BOX[13]],
    26: [x*y for x, y in ANCHOR_BOX[26]],
    52: [x*y for x, y in ANCHOR_BOX[52]]
                }

COCO_CLASS = ["person", "bicycle", "car", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant"]






