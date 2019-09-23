# -*- coding:utf-8 -*-
import torchvision
import torchvision.transforms as tf


# draw
CLS = 10
FEATURE = 2
COLOR = {
    0: "green", 1: "red", 2: "blue", 3: "black", 4: "purple", 5: "gray", 6: "gold", 7: "m", 8: "pink", 9: "peru"
        }


# download 
PATH = "F:/Mnist"
train_data = torchvision.datasets.MNIST(root=PATH, train=True, transform=tf.ToTensor(), download=False)
test_data = torchvision.datasets.MNIST(root=PATH, train=False, transform=tf.ToTensor(), download=False)


