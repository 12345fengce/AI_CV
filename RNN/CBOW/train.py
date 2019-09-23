# -*- coding:utf-8 -*-
import os
import torch
import cbownet
import dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, params: str):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = cbownet.CBOW().to(self.device)
        self.summarywriter = SummaryWriter(log_dir="./runs")
        if os.path.exists(params):
            self.net.load_state_dict(torch.load(params))
        self.train = data.DataLoader(dataset.mydata, batch_size=128, shuffle=True)
        self.optimize = optim.Adam(self.net.parameters())
        self.loss = nn.MSELoss()

    def main(self):
        for epoche in range(5000):
            self.net.train()
            for i, (sentence, x) in enumerate(self.train):
                x = x.to(self.device)
                y = self.net(x)
                t = self.net.lexicon(x[:, 2])
                loss = self.loss(y, t)

                self.optimize.zero_grad()  # backward
                loss.backward()
                self.optimize.step()

                print("[epoche - {}] - {}/{} - Loss: {}".format(epoche, i, len(self.train), loss.item()))
            self.summarywriter.add_scalar("Loss", loss.item(), global_step=epoche)
            torch.save(self.net.state_dict(), self.params)


if __name__ == '__main__':
    params = "./cbow.pkl"
    mytrain = Train(params)
    mytrain.main()
