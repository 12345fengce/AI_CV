# -*-coding:utf-8-*-
import os
import torch
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf




class MyData(data.Dataset):
    def __init__(self, path, size):
        super(MyData, self).__init__()
        self.path = path
        self.size = size
        self.targets = []
        # 将同一size图片，标签集中起来，放到两个列表里
        for direc in ["positive", "negative", "part"]:
            _path = os.path.join(path, str(size), direc)
            if not os.path.exists(_path):
                continue
            with open(_path+".txt", "r") as f:
                self.targets.extend(f.readlines())

    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, index):
        line = self.targets[index].split()
        name, confidence, coordinate, landmark = line
        confi = torch.Tensor(np.array([confidence], dtype=np.int))
        offset = torch.Tensor(np.array(eval(coordinate)+eval(landmark), dtype=np.float))
        _path = self.path+"/"+str(self.size)+"/"+name
        _img = Image.open(_path)
        img = tf.ToTensor()(_img)-0.5
        return img, confi, offset


if __name__ == '__main__':
    path = "G:/for_MTCNN/train"
    size = 12
    mydata = MyData(path, size)
    print(mydata[100], len(mydata.targets))


            



