# -*- coding:utf-8 -*-
import os
import cfg
import torch
import numpy as np
import torch.nn as nn
import scipy.io.wavfile as wav
import torch.utils.data as data


class MyData(data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.filebase = []
        for dir in os.listdir(path):
            dir = path+"/"+dir
            for file in os.listdir(dir):
                file = dir+"/"+file
                self.filebase.append(file)

    def __len__(self):
        return len(self.filebase)

    def __getitem__(self, index):
        file = self.filebase[index]
        label = torch.tensor(int(file.split("/")[-2]))
        sample_rate, data = wav.read(file)
        data = torch.Tensor(data.transpose(1, 0))
        return data, label


TRAIN = data.DataLoader(MyData(cfg.TRAIN_PATH), batch_size=cfg.BATCH, shuffle=True)
TEST = data.DataLoader(MyData(cfg.TEST_PATH), batch_size=cfg.BATCH, shuffle=True, drop_last=True)


