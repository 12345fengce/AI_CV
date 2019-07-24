# -*-coding:utf-8-*-
import torch
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as tf


# draw
color = {
    0: "green", 1: "red", 2: "blue", 3: "black", 4: "purple", 5: "gray", 6: "gold", 7: "m", 8: "pink", 9: "peru"
        }

# if __name__ == "__main__":
#     index = torch.tensor([0])
#     print(color[torch.tensor([0])])
#     index = np.array([1, 2, 3])
#     for i in index:
#         print(color[i])


# download 
path = "G:/Mnist"
train_data = torchvision.datasets.MNIST(root=path, train=True, transform=tf.ToTensor(), download=False)
test_data = torchvision.datasets.MNIST(root=path, train=False, transform=tf.ToTensor(), download=False)

# if __name__ == "__main__":
#     data = train_data.data
#     label = train_data.targets
#     print(data.size(), label.size())
#     train = data.DataLoader(train_data, batch_size=512, shuffle=True)
#     for i, t in train:
#         print(i.size())
