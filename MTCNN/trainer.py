# -*-coding:utf-8-*-
import os, sys
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, net, data_path, save_path, img_size):
        # cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # path
        self.data_path = data_path
        self.save_path = save_path
        self.img_size = img_size
        # net
        if os.path.exists(save_path):
            self.net = torch.load(save_path)
        else:
            self.net = net.to(self.device)
        # dataset
        self.data = data.DataLoader(dataset.MyData(data_path, img_size), batch_size=512, shuffle=True, num_workers=3)
        # optim
        self.opt = optim.SGD(self.net.parameters(), lr=0.0009)
        # loss
        self.loss_confi = nn.BCELoss()
        self.loss_offset = nn.MSELoss()

    def train(self):
        epoche = 1
        loss1 = []
        loss2 = []
        while True:
            print("[epoche] - {}:".format(epoche))
            for i, (x, confi, offset) in enumerate(self.data):
                # 将数据加载到GPU上
                x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
                # 输入→输出
                c, o = self.net(x)
                # PNet与RNet、ONet输出形状不一，在这里统一处理
                c, o = c.view(-1, 1).to(self.device), o.view(-1, 4).to(self.device)
                # 剔除相应数据，置信度采用positive和negative，偏移量采用positive和part
                o, offset = o[confi.view(-1) != 0], offset[confi.view(-1) != 0]
                c, confi = c[confi.view(-1) != 2], confi[confi.view(-1) != 2]
                # 计算损失
                loss_c = self.loss_confi(c, confi)
                loss_o = self.loss_offset(o, offset)
                loss = loss_c+loss_o
                # 损失梯度下降
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # 输出每轮损失，保存网络
            with torch.no_grad():
            epoche += 1
            if epoche % 10 == 0:
                torch.save(self.net, self.save_path)



            



        


