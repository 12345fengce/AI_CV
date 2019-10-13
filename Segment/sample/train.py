# -*- coding:utf-8 -*-
import os
import sys
import cfg
import net
import time
import utils
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import PIL.Image as Image


class MyTrain:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        self.net = net.UNetPlusPlus(cfg.CLS_NUM).to(self.device)
        if not os.path.exists(cfg.PARAMS):
            print("Initing ... ...")
            self.net.apply(utils.weights_init)
        else:
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(cfg.PARAMS))
        # Data
        self.train = dataset.TRAIN
        # Loss
        self.loss = nn.BCELoss(reduction='sum')
        # Optimize
        self.opt = optim.Adam(self.net.parameters(), lr=1e-2)

    def run(self, log: str, lower_loss=17000):
        with open(log, "a+") as f:
            # Configure Written
            f.write("\n{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(">>> The Size of Input Images: {}\n".format(cfg.SIZE))
            # Train
            for epoch in range(cfg.EPOCH):
                f.write(">>> epoch: {}\n".format(epoch))
                self.net.train()
                loss_list = []
                for i, (x, t) in enumerate(self.train):
                    x, t = x.to(self.device), (utils.separate(t, cfg.CLS_NUM)).to(self.device)
                    output = self.net(x)
                    loss = self.loss(output, t)
                    # Backward
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    loss_list.append(loss.item())
                    print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))
                loss_mean = sum(loss_list)/len(loss_list)
                f.write(">>> Loss: {}\n".format(loss_mean))
                Save
                if loss_mean < lower_loss:
                    lower_loss = loss_mean
                    f.write(">>> SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                    torch.save(self.net.state_dict(), cfg.PARAMS)
                f.flush()


if __name__ == '__main__':
    log = "./log.txt"
    MyTrain().run(log)

