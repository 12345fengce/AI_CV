# -*- coding:utf-8 -*-
import os
import cfg
import utils
import torch
import torch.nn as nn
import PIL.Image as Image
import torch.utils.data as data


class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        self.path = path
        self.database = []
        for file in os.listdir(path+"/labels"):
            if "regions" in file:
                self.database.append(file)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        file = self.database[index]
        imgname = file.split(".")[0]+".jpg"
        imgpath = self.path+"/images/"+imgname
        img = utils.transform(Image.open(imgpath))

        labelfile = self.path+"/labels/"+file
        with open(labelfile, "r") as f:
            labellist = f.readlines()
        label = []
        for labelstr in labellist:
            labelstr = labelstr.replace(" ", ",").replace("\n", "")
            label.append(torch.tensor(eval(labelstr), dtype=torch.int8))
        label = torch.stack(label, dim=0)

        normalize = nn.AdaptiveAvgPool2d((cfg.SIZE[1], cfg.SIZE[0]))
        _, h, w = img.size()
        if w != cfg.SIZE[0] or h != cfg.SIZE[1]:
            if w < h:
                img = img.permute(0, 2, 1)
                label = label.permute(1, 0)
            img = normalize(img)
            label = normalize(label.unsqueeze(dim=0).float())

        return img, label.long().squeeze()


TRAIN = data.DataLoader(dataset=MyData(cfg.PATH), batch_size=cfg.BATCH, shuffle=True, num_workers=4)



