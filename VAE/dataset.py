# -*- coding:utf-8 -*-
from torchvision import transforms, datasets


Z = 10
TRANSFROM = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))
                                ])
TRAIN = datasets.MNIST(root="F:/Mnist", train=True, transform=TRANSFROM, download=False)
TEST = datasets.MNIST(root="F:/Mnist", train=False, transform=TRANSFROM, download=False)


# if __name__ == '__main__':
#     train = TRAIN.train_data
#     print(train.shape)