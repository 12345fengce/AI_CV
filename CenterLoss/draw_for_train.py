# -*-coding:utf-8-*-
import os
import cfg
import net
import time
import utils
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class MyTrain:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        self.net = net.MyNet().to(self.device)
        if not os.path.exists(cfg.PATH) or not os.path.exists(cfg.IMG):
            os.makedirs(cfg.PATH)
            os.makedirs(cfg.IMG)
        if not os.path.exists(cfg.MODEL):
            print("Initing ... ...")
            self.net.apply(utils.weights_init)
        else:
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(cfg.MODEL))
        # Data
        self.train = dataset.TRAIN
        self.test = dataset.TEST
        # Loss
        self.centerloss = net.CenterLoss(cfg.CLS_NUM, cfg.FEATURE_NUM).to(self.device)
        if os.path.exists(cfg.CLOSS):
            self.centerloss.load_state_dict(torch.load(cfg.CLOSS))
        self.clsloss = nn.CrossEntropyLoss()
        # Optimize
        self.opt = optim.Adam(self.net.parameters())
        self.opt_centerloss = optim.SGD(self.centerloss.parameters(), lr=0.5)

    def run(self):
        # Train
        for epoch in range(cfg.EPOCH):
            self.net.train()
            coordinate, target = [], []
            for i, (x, t) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                features, outputs = self.net(x)
                loss_center = self.centerloss(features, t)
                loss_cls = self.clsloss(outputs, t)
                loss = loss_cls+cfg.LAMBDA*loss_center
                # Backward
                self.opt.zero_grad()
                self.opt_centerloss.zero_grad()
                loss.backward()
                self.opt.step()
                self.opt_centerloss.step()

                target.extend(t)
                coordinate.extend(features)

                print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))

            coordinate, target = torch.stack(coordinate).cpu().data.numpy(), torch.stack(target).cpu().data.numpy()
            plt.clf()
            for num in range(10):
                plt.scatter(coordinate[target == num, 0], coordinate[target == num, 1], c=cfg.COLOR[num], marker=".")
            plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
            center = self.centerloss.center.data.cpu().numpy()
            plt.scatter(center[:, 0], center[:, 1], c=cfg.COLOR[10])
            plt.title("[epoch] - {}".format(epoch), loc="left")
            plt.savefig("{}/pic{}.png".format(cfg.IMG, epoch))

if __name__ == '__main__':
    MyTrain().run()
