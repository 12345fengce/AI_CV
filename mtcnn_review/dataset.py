# -*- coding:utf-8 -*-
import os
import cfg
import torch
import random
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf


def gen(data_path, label_path, landmark_path, save_path):
    for size in cfg.NET.values():
        "创建文件夹"
        part = data_path+"/"+str(size)+"/part"
        negative = data_path + "/" + str(size) + "/negative"
        positive = data_path + "/" + str(size) + "/positive"
        for path in [positive, negative, part]:
            if not os.path.exists(path):
                os.makedirs(path)
        part_label = open(part+".txt", "a")
        positive_label = open(positive+".txt", "a")
        negative_label = open(negative+".txt", "a")
        "读取标签信息"
        with open(label_path) as f:
            targets = f.readlines()
        with open(landmark_path) as f:
            landmarks = f.readlines()
        count = 0
        for i in range(len(targets)):
            if i < 2:
                continue
            filename, *coordinate = offsets[i].split()
            coordinate = np.array(coordinate, dtype=np.int)
            landmark = np.array(landmarks[i].split()[1:], dtype=np.float)
            if np.sum(coordinate < 0) < coordinate.size:
                continue
            image = Image.open(data_path+"/"+filename)
            m = min(image.size)
            mx, my = coordinate[0]+coordinate[2]/2, coordinate[1]+coordinate[3]/2
            "循环生成新图片"
            j = 0
            while j < 3:
                x_offset, y_offset = random.uniform(-0.3*m, 0.3*m), random.uniform(-0.3*m, 0.3*m)
                side = random.uniform(0.6*m, 1.2*m)









class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
