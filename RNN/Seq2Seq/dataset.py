# -*-coding:utf-8-*-
import os
import torchvision
import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as tf


class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        self.database = os.listdir(path)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        img_name = self.database[index]
        img_path = self.path+"/"+img_name
        img = Image.open(img_path)
        img = tf.ToTensor()(img)
        return img


TRAIN = torchvision.datasets.MNIST(root="F:/Mnist", train=True, transform=tf.ToTensor(), download=False)
TEST = torchvision.datasets.MNIST(root="F:/Mnist", train=False, transform=tf.ToTensor(), download=False)
# if __name__ == '__main__':
#     path = "F:/MTCNN/train/48/positive"
#     mydata = MyData(path)
#     print(mydata[0])