# -*- coding:utf-8 -*-
import os
import cfg
import utils
import torch
import numpy as np
import torch.nn as nn
import PIL.Image as Image
import torch.utils.data as data


class MyData(data.Dataset):
    def __init__(self, path, type):
        super(MyData, self).__init__()
        self.path = path
        self.database = []
        self.label = []
        for dir in os.listdir(path):
            print(dir)
            filepath = path+"/"+dir+"/"+type
            for i, file in enumerate(os.listdir(filepath)):
                if "png" in file:
                    labelpath = filepath+"/"+file
                    array = np.array(Image.open(labelpath))
                    if np.sum(array) > 0:
                        self.label.append(array)
                        imgpath = filepath + "/" + os.listdir(filepath)[i - 1]
                        print(utils.dcm2png(imgpath))
                        self.database.append(utils.dcm2png(imgpath))
                else:
                    continue

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        img = (torch.Tensor(self.database[index])/255-0.5)/0.5
        label = torch.tensor(self.label[index].transpose(2, 0, 1))
        return img, label


if __name__ == '__main__':
    path = "F:/Segment/B_task/data"
    type = "arterial phase"
    mydata = MyData(path, type)
    img, label = mydata[0]
    print(img.shape, label.shape)
    print(img)
    print(label)



