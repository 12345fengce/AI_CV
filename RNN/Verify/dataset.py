# -*- coding:utf-8 -*-
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as Draw
import PIL.ImageFont as Font

import os
import torch
import torch.utils.data as data
import torchvision.transforms as tf


LEVEL = 5
SIZE = (5000, 60, 120, 3)
SAVE = "F:/RNN/VERIFY/train"
FONT = "C:/Windows/Fonts/FREESCPT.TTF"


class Generate:
    def __init__(self, save: str, size: tuple):
        """save: the path of imgs saved
            size: (img_num, size of fingure(h, w), channels_num)"""
        self.size = size
        self.save = save
        self.background = np.random.randint(0, 150, size, dtype=np.uint8)  # pixes's value range of figure

    def gen(self, font: str, level: int):
        """font: font of nums
            level: amount of nums"""
        font = Font.truetype(font, 40)
        for i in range(self.size[0]):
            text = ""
            background = Image.fromarray(self.background[i])
            draw = Draw.Draw(background, "RGB")
            for j in range(level):
                color = tuple(np.random.randint(120, 255, 3))  # pixes's value range of nums
                txt = chr(np.random.randint(48, 58))  # ascii of 0-9
                text += str(txt)
                x, y = 20*j, np.random.randint(0, 20)  # coordinates of nums
                draw.text((x, y), txt, color, font)
            background.save(self.save+"/{}.jpg".format(text))
            if i % 100 == 0:
                print("I'm runing, {} has Done!!!".format(i))


class MyData(data.Dataset):
    def __init__(self, path: str):
        super(MyData, self).__init__()
        self.path = path
        self.database = os.listdir(path)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        label = torch.tensor(np.array(list(self.database[idx].split(".")[0]), dtype=np.int))
        img_path = self.path+"/"+self.database[idx]
        img = Image.open(img_path)
        img = tf.ToTensor()(img)-0.5

        return img, label


if __name__ == '__main__':
#
#     generation = Generate(SAVE, SIZE)
#     generation.gen(FONT, LEVEL)

    PATH = "F:/RNN/Verify/test"
    mydata = MyData(PATH)
    print(mydata[-1])