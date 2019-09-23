# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from RNN.Seq2Seq import dataset as D
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, params: str):
        self.params = params
        self.device = torch.device("cuda")
        self.net = net.HighLight().to(self.device)
        self.summarywriter = SummaryWriter(log_dir="G:/Project/Code/RNN/Attention/runs")
        if os.path.exists(params):
            self.net.load_state_dict(torch.load(params))
        self.train = data.DataLoader(dataset.MyData("F:/MTCNN/validation/48"), batch_size=128, shuffle=True, num_workers=4)
        self.test = data.DataLoader(D.MyData("F:/RNN/ATTEN/test"), batch_size=256, shuffle=True)
        self.optimize = optim.Adam(self.net.parameters(), weight_decay=0.001)
        self.loss = nn.BCELoss()

    def main(self):
        for epoche in range(5000):
            self.net.train()
            for i, (x, t) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                new, confi = self.net(x)
                loss = self.loss(confi, t)

                self.optimize.zero_grad()
                loss.backward()
                self.optimize.step()

                # plt.clf()
                # plt.suptitle("epoche: {} Loss: {:.5f}".format(epoche, loss.item()))
                # idx, title = 1, "origin"
                # for j in range(1, 5):
                #     if j > 2:
                #         x = new  # 原图片与生成图片切换
                #         title = "new"
                #     img = x.permute(0, 2, 3, 1).data.cpu().numpy()
                #     plt.subplot(2, 2, j)
                #     plt.axis("off")
                #     plt.title(title)
                #     plt.imshow(np.uint8(img[idx]*255))
                #     idx *= -1
                # plt.pause(0.1)
                self.summarywriter.add_scalar("Loss", loss.item(), global_step=i)
            torch.save(self.net.state_dict(), self.params)
            with torch.no_grad():
                self.net.eval()
                rec = []
                for _x in self.test:
                    _new, _confi = self.net(_x.cuda())
                    print(_confi[-1].item())
                    rec.append(torch.sum(_confi).item())
                acc = sum(rec)/12719
                self.summarywriter.add_scalar("REC", acc, global_step=epoche)
                print("MODE: Verify  REC: ", acc)


if __name__ == '__main__':
    params = "G:/Project/Code/RNN/Attention/params.pkl"
    mytrain = Train(params)
    mytrain.main()





