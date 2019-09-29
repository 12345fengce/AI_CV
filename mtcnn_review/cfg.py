# -*- coding:utf-8 -*-

"Global Variables"
PATH = "F:/mtcnn_v1/train"
EPOCH = 33333

"iou to identify positive and part"
INTERVAL = {
        "positive": 0.6,
        "part": 0.3,
                }

"size of pictures for net"
NET = {
        "pnet": [12, 2],
        "rnet": [24, 4],
        "onet": [48, 8],
        }

"batch_size"
BATCH = {
        "pnet": 64,
        "rnet": 64,
        "onet": 128,
        }
TEST_BATCH = 512

"the weights of kind of loss"
LOSS = {
        "pnet": [1, 0.5, 0.5],
        "rnet": [1, 0.5, 0.5],
        "onet": [1, 0.5, 1],
        }

"the threshold of confidence"
CONFI = {
        "pnet": 0.9,
        "rnet": 0.99,
        "onet": 0.999,
        }

LOWER_ERROR = {
        "pnet": 0.1,
        "rnet": 0.3,
        "onet": 0.5,
        }



