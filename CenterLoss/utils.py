# -*- coding:utf-8 -*-
import imageio
from torch.nn import init


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "Linear"]:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
    elif classname.find("PReLU") != -1:
        init.constant_(m.weight.data, 0.01)


def gif(path: str, lambd: int, fps: int):
    gif_list = []
    for i in range(305):
        file = path+"/pic{}.png".format(i)
        gif_list.append(imageio.imread(file))
    imageio.mimsave(path+"/lambda_{}.gif".format(lambd), gif_list, fps=fps)

if __name__ == '__main__':
    path = "G:/Project/Code/CenterLoss/img"
    gif(path, 10, 10)

