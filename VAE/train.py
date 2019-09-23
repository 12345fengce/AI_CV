# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, params: str):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.MainNet().to(self.device)
        self.summarywriter = SummaryWriter(log_dir="./runs")
        if os.path.exists(params):
            self.net.load_state_dict(torch.load(params))
        self.train = data.DataLoader(dataset.TRAIN, batch_size=100, shuffle=True)
        self.optimize = optim.Adam(self.net.parameters())
        self.loss_refactor = nn.MSELoss(reduction="sum")

    def main(self, alpha=0.1):
        for epoche in range(5000):
            self.net.train()
            for i, (x, t) in enumerate(self.train):
                x = x.to(self.device)
                mean, var, xs = self.net(x)

                loss_kl = 0.5*torch.sum(mean**2+var-torch.log(var)-1)
                loss_refactor = self.loss_refactor(xs, x)
                loss = alpha*loss_kl+(1-alpha)*loss_refactor

                self.optimize.zero_grad()  # backward
                loss.backward()
                self.optimize.step()

                print("[epoche] - {} - {}/{}\nloss_kl:{}\nloss_refactor:{}\nLoss:{}".format(epoche, i, len(self.train),
                                                                                            loss_kl.item(),
                                                                                            loss_refactor.item(),
                                                                                            loss.item()))
                self.summarywriter.add_scalar("Loss", loss.item(), global_step=i)
            torch.save(self.net.state_dict(), self.params)
            save_image(xs, "./img/re{}.png".format(epoche), nrow=10, normalize=True, scale_each=True)
            save_image(x, "./img/tr{}.png".format(epoche), nrow=10, normalize=True, scale_each=True)


if __name__ == '__main__':
    params = "G:/Project/Code/VAE/mainnet.pkl"
    mytrain = Train(params)
    mytrain.main()