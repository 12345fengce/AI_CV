# -*-coding:utf-8-*-
import os
import cfg
import utils
import torch
import PIL.Image as Image
import torch.utils.data as data


class MyData(data.Dataset):
    def __init__(self, path, train=True):
        super(MyData, self).__init__()
        self.path = path
        self.database = []
        for dir in os.listdir(path):
            img_path = path+"/"+dir
            for i, img_name in enumerate(os.listdir(img_path)):
                if train:
                    if i < 4:
                        self.database.append(img_name)
                    continue
                else:
                    if i == 4:
                        self.database.append(img_name)
                    continue

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        img_name = self.database[index]
        label = img_name.split("_")[0]
        img = self.path+"/"+label+"/"+img_name

        img = utils.transform(Image.open(img))
        label = torch.tensor(int(label))
        return img, label


TRAIN = data.DataLoader(MyData(cfg.PATH, train=True), batch_size=cfg.BATCH, shuffle=True)
TEST = data.DataLoader(MyData(cfg.PATH, train=False), batch_size=cfg.BATCH, shuffle=True)
