# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, params: str):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.MainNet().to(self.device)
        self.summarywriter = SummaryWriter(log_dir="./runs")
        if os.path.exists(params):
            self.net.load_state_dict(torch.load(params))
        self.train = data.DataLoader(dataset.TRAIN, batch_size=8, shuffle=True, drop_last=True)
        self.test = data.DataLoader(dataset.TEST, batch_size=512, shuffle=True)
        self.optimize = optim.Adam(self.net.parameters())
        self.loss = nn.MSELoss()

    def main(self):
        for epoche in range(5000):
            self.net.train()
            for i, (x, t) in enumerate(self.train):
                x = x.to(self.device)
                y = self.net(x)
                loss = self.loss(y, x)
                self.optimize.zero_grad()  # backward
                loss.backward()
                self.optimize.step()
                plt.clf()                  # draw
                plt.suptitle("epoche: {} Loss: {:.5f}".format(epoche, loss.item()))
                idx, title = 1, "origin"
                for j in range(1, 5):
                    if j > 2:
                        x = y
                        title = "new"
                    img = x.data.permute(0, 2, 3, 1).cpu().numpy()
                    plt.subplot(2, 2, j)
                    plt.axis("off")
                    plt.title(title)
                    plt.imshow(np.uint8(255*img[idx]).reshape(28, 28))
                    idx *= -1
                plt.pause(0.1)
                print("{}/{}".format(i, len(self.train)))
                self.summarywriter.add_scalar("Train Loss", loss.item(), global_step=i)
            torch.save(self.net.state_dict(), self.params)
            with torch.no_grad():
                self.net.eval()
                for _x, _t in self.test:
                    _x = _x.to(self.device)
                    _y = self.net(_x)
                    _loss = self.loss(_y, _x)
                    self.summarywriter.add_scalar("Test Loss", _loss.item(), global_step=epoche)
                    break


if __name__ == '__main__':
    params = "G:/Project/Code/RNN/mainnet.pkl"
    mytrain = Train(params)
    mytrain.main()





