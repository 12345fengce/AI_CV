# -*- coding:utf-8 -*-
import torchvision
import torch.utils.data as data
import torchvision.transforms as tf


datapath = "F:/Mnist"
transform = tf.Compose([
                        tf.ToTensor(),
                        tf.Normalize(mean=(0.5,), std=(0.5,))
                        ])
train_data = torchvision.datasets.MNIST(root=datapath, train=True, transform=transform, download=False)
test_data = torchvision.datasets.MNIST(root=datapath, train=False, transform=transform, download=False)

train = data.DataLoader(train_data, batch_size=512, shuffle=True, num_workers=4)
test = data.DataLoader(test_data, batch_size=1024, shuffle=False)


# if __name__ == '__main__':
#     for i, (x, t) in enumerate(test):
#         print(i)
#         print(x.size())
#         print(t[7])
#         break