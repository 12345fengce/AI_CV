# -*-coding:utf-8-*-
import os
import cfg
import utils
import torch
import numpy as np
import PIL.Image as Image
import torch.utils.data as data


def gen(data: str, label: str, save: str):

    with open(label, "r") as f:
        lines = f.readlines()
    txt = open(save+".txt", "w")

    for i, line in enumerate(lines):
        if len(line.split()) == 1:
            continue
        filename, *boxes = line.split()
        filepath = data+"/"+filename
        image = Image.open(filepath)
        image = image.resize((416, 416))

        boxes = np.array(boxes, dtype=np.int)
        boxes = np.split(boxes, len(boxes)//5)

        bbox = []
        for box in boxes:
            cls, cx, cy, w, h = box
            if cls < 10:
                bbox.extend([cls, cx, cy, w, h])
            else:
                continue
        if len(bbox) != 0:
            image.save(save+"/"+filename)
            txt.write("{} {}\n".format(filename, bbox))

    
class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        label_path = path+".txt"
        with open(label_path) as f:
            self.label = f.readlines()

    def __len__(self):
        return len(self.label)

    def __getlabel__(self, data):
        label = {}
        for key, item in cfg.ANCHOR_BOX.items():
            label[key] = np.zeros(shape=(key, key, len(item), 5+1))
            for box in data:
                cls, x, y, w, h = box
                "Center offset"
                (x_offset, x_index), (y_offset, y_index) = np.modf(x*key/cfg.IMG_SIZE), np.modf(y*key/cfg.IMG_SIZE)
                for i, anchor in enumerate(item):
                    "W, H offset"
                    w_offset, h_offset = np.log(w/anchor[0]), np.log(h/anchor[1])  # b/p
                    "Iou"
                    inter_area = min(w, anchor[0]) * min(h, anchor[1])
                    union_area = w * h + cfg.ANCHOR_AREA[key][i] - inter_area
                    iou = inter_area / union_area
                    "Lable"
                    label[key][int(x_index), int(y_index), i] = \
                        np.array([iou, x_offset, y_offset, w_offset, h_offset, int(cls)])
        return label

    def __getitem__(self, index):
        img_path, *data = self.label[index].split(".jpg ")
        img_path = self.path+"/"+img_path+".jpg"
        img = Image.open(img_path)
        img = utils.transform(img)
        data = np.array(eval(data[0]), dtype=np.float)
        data = np.array(np.split(data, len(data)//5))
        label_13, label_26, label_52 = self.__getlabel__(data).values()
        return torch.Tensor(label_13), torch.Tensor(label_26), torch.Tensor(label_52), img



