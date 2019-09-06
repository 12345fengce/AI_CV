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
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, params: str, inputs: str):
        self.params = params
        self.device = torch.device("cuda")
        self.net = net.MainNet().to(self.device)
        self.summarywriter = SummaryWriter(log_dir="G:/Project/Code/RNN/Seq2Seq/runs")
        if len(os.listdir(params)) > 2:
            self.net.extractor.load_state_dict(torch.load(params+"/extractor.pkl"))
            self.net.generator.load_state_dict(torch.load(params+"/generator.pkl"))
        self.train = data.DataLoader(dataset.MyData(inputs), batch_size=128, shuffle=True, num_workers=4)
        self.optimize = optim.Adam([{"params": self.net.extractor.parameters()},
                                    {"params": self.net.generator.parameters()}])
        self.loss = nn.MSELoss()

    def main(self):
        for epoche in range(100):
            self.net.train()
            for i, x in enumerate(self.train):
                x = x.to(self.device)
                y = self.net(x)
                loss = self.loss(y, x)

                self.optimize.zero_grad()
                loss.backward()
                self.optimize.step()

                # plt.clf()
                # plt.suptitle("epoche: {} Loss: {:.5f}".format(epoche, loss.item()))
                # idx, title = 1, "origin"
                # for j in range(1, 5):
                #     if j > 2:
                #         x = y
                #         title = "new"
                #     plt.subplot(2, 2, j)
                #     plt.axis("off")
                #     plt.title(title)
                #     plt.imshow(self.toimg(x[idx]))
                #     idx *= -1
                # plt.pause(0.1)
                self.summarywriter.add_scalar("Loss", loss.item(), global_step=i)
            print("epoche", epoche)
            torch.save(self.net.extractor.state_dict(), self.params+"/extractor.pkl")
            torch.save(self.net.generator.state_dict(), self.params+"/generator.pkl")

    def toimg(self, x):
        normalize_data = x.data.cpu().numpy()
        data = normalize_data.transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(data*255))
        return img


if __name__ == '__main__':
    params = "G:/Project/Code/RNN/Seq2Seq/params"
    inputs = "F:/MTCNN/validation/48/positive"
    mytrain = Train(params, inputs)
    mytrain.main()





