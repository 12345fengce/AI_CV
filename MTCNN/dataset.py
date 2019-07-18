# -*-coding:utf-8-*-
import os
import torch
import torch.utils.data as data
import torchvision.transforms as tf
import PIL.Image as Image
import numpy as np 



class MyData(data.Dataset):
    def __init__(self, path, img_size):
        super(MyData, self).__init__()
        self.path = path
        self.img_size = img_size
        self.dataset = []
        self.targets = []
        # 将同一size图片，标签集中起来，放到两个列表里
        for dir in ["positive", "negative", "part"]:
            _path = os.path.join(path, str(img_size), dir)
            if not os.path.exists(_path):
                continue
            self.dataset.extend(os.listdir(_path))
            with open(_path+".txt", "r") as f:
                self.targets.extend(f.readlines())

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        line = self.targets[index].split()
        img_file, confidence, x1, y1, x2, y2 = line
        confi = torch.Tensor(np.array([confidence], dtype=np.int))  
        offset = torch.Tensor(np.array([x1, y1, x2, y2], dtype=np.float))  
        img_path = self.path+"/"+str(self.img_size)+"/"+img_file
        img = Image.open(img_path)
        data = tf.ToTensor()(img)-0.5
        return data, confi, offset



            



