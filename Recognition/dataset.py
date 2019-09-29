# -*- coding:utf-8 -*-
import cfg
import torchvision
from torch.utils import data
import torchvision.transforms as tf


path = "F:/Mnist"
transform = tf.Compose([
                        tf.ToTensor(),
                        tf.Normalize((0.5, ), (0.5, )),
                        ])

train_data = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=False)
test_data = torchvision.datasets.MNIST(root=path, train=False, transform=transform, download=False)

TRAIN = data.DataLoader(dataset=train_data, batch_size=cfg.BATCH, shuffle=True, drop_last=True)
TEST = data.DataLoader(dataset=test_data, batch_size=cfg.BATCH, shuffle=True, drop_last=True)


