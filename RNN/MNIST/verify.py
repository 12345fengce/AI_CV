# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import PIL.Image as Image
import torchvision.transforms as tf


def verify(imgs, parameter):
    model = net.RnnNet().cuda()
    model.load_state_dict(torch.load(parameter))
    model.eval()
    for path in os.listdir(imgs):
        img = imgs+"/"+path
        img = Image.open(img)
        img = img.convert("L")
        img = img.resize((28, 28))

        img = tf.ToTensor()(img).unsqueeze(dim=0)-0.5
        output = model(img.cuda())
        predict = torch.argmax(output, dim=1)
        print("Truth:{}\nPredict:{}".format(path[0], predict.item()))


if __name__ == '__main__':
    imgs = "F:/RNN/MNIST"
    parameter = "G:/Project/Code/RNN/MNIST/rnn.pt"
    verify(imgs, parameter)



