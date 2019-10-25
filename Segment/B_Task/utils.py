# -*- coding:utf-8 -*-
import torch
import SimpleITK as sitk
import torch.nn.init as init
import torchvision.transforms as tf


def dcm2png(img):
    """(W, H, C) → (C, H, W)"""
    img = sitk.ReadImage(img)
    array = sitk.GetArrayFromImage(img)
    return array


def transform(img):
    operate = tf.Compose([
                            tf.ToTensor(),
                            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
    return operate(img)


def separate(label, cls_num):
    """label: (N, W, H)"""
    labels = torch.zeros((label.size(0), cls_num, label.size(1), label.size(-1)))
    for i in range(cls_num):
        mask = (label == i)
        labels[:, i, :, :][mask] += 1
    return labels


def fill(output, cls_num, cls_color, threshold):
    """output: (N, C, H, W)"""
    image = torch.zeros(output.size(0), 3, output.size(2), output.size(3))
    for i in range(cls_num):
        mask = (output[:, i, :, :] >= threshold)
        for j, pixel in enumerate(cls_color[i]):
            image[:, j, :, :][mask] += pixel
    return image


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "Linear"]:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
    elif classname.find("PReLU") != -1:
        init.constant_(m.weight.data, 0.01)


if __name__ == '__main__':
    path = "F:/Segment/B_task/data/1001/arterial phase/10001.dcm"
    array = dcm2png(path)
    print(array)