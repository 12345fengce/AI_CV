# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data
        self.train = data.DataLoader(dataset.mydata, batch_size=dataset.BATCH_SIZE, shuffle=True, drop_last=True)
        # net
        self.generator = net.GenNet().apply(net.weights_init).to(self.device)
        self.arbiter = net.ArbiNet().apply(net.weights_init).to(self.device)
        if len(os.listdir("./params")) > 1:
            print("Loading ... ...")
            self.generator.load_state_dict(torch.load("./params/gen.pkl"))
            self.arbiter.load_state_dict(torch.load("./params/ari.pkl"))
        # optimize
        self.g_opt = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.a_opt = optim.Adam(self.arbiter.parameters(), lr=2e-4, betas=(0.5, 0.999))
        # loss
        self.loss = nn.BCELoss()
        # show
        self.write = SummaryWriter("./runs")

    def main(self):
        for epoche in range(150):
            print("epoche >>> {}".format(epoche))
            for i, tx in enumerate(self.train):
                # label
                tl = torch.ones((dataset.BATCH_SIZE, 1, 1, 1)).to(self.device)
                fl = torch.zeros((dataset.BATCH_SIZE, 1, 1, 1)).to(self.device)
                # arbiter data
                tx = tx.to(self.device)
                fx = torch.randn((dataset.BATCH_SIZE, 512, 1, 1)).to(self.device)
                fx = self.generator(fx)
                # arbiter forward
                t_output = self.arbiter(tx)
                f_output = self.arbiter(fx)
                # arbiter loss
                a_loss = self.loss(t_output, tl)+self.loss(f_output, fl)
                # arbiter backward
                self.a_opt.zero_grad()
                a_loss.backward()
                self.a_opt.step()

                # generate data
                gx = torch.randn((dataset.BATCH_SIZE, 512, 1, 1)).to(self.device)
                # generate forward
                gx = self.generator(gx)
                g_output = self.arbiter(gx)
                # generate loss
                g_loss = self.loss(g_output, tl)
                # generate backward
                self.g_opt.zero_grad()
                g_loss.backward()
                self.g_opt.step()
                # show
                print("a_score : {} g_score : {}".format(t_output.mean(), g_output.mean()))
                self.write.add_scalar("a_loss", a_loss.item(), i)
                self.write.add_scalar("g_loss", g_loss.item(), i)

            # save
            torch.save(self.arbiter.state_dict(), "./params/ari.pkl")
            torch.save(self.generator.state_dict(), "./params/gen.pkl")
            save_image(tx.cpu().data, "./img/True_{}.png".format(epoche), nrow=8)
            save_image(gx.cpu().data, "./img/False_{}.png".format(epoche), nrow=8)


if __name__ == '__main__':
    # pretreat
    if not os.path.exists("./img") or not os.path.exists("./params"):
        os.makedirs("./img")
        os.makedirs("./params")
    mytrain = Trainer()
    mytrain.main()