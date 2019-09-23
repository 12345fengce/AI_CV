# -*- coding:utf-8 -*-
import os
import torch
import torchvision
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf


BATCH_SIZE = 64
PATH = "F:/GAN/faces"
transform = tf.Compose([
                        tf.ToTensor(),
                        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

transforms = tf.Compose([
                        tf.ToTensor(),
                        tf.Normalize((0.5,), (0.5,))
                        ])


TRAIN = torchvision.datasets.MNIST("F:/Mnist", train=True, transform=transforms, download=False)
TEST = torchvision.datasets.MNIST("F:/Mnist", train=False, transform=transforms, download=False)


class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        self.database = os.listdir(path)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        img_path = self.path+"/{}".format(self.database[idx])
        img = Image.open(img_path)
        x = transform(img)
        return x



mydata = MyData(PATH)
# img = mydata[0]
# print(img)
