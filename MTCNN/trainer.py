# -*-coding:utf-8-*-
import os
import torch
import random
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, net, train_path, validation_path, save_path, img_size):
        # cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # path
        self.save_path = save_path
        self.img_size = img_size
        # net
        if os.path.exists(save_path):
            self.net = torch.load(save_path)
        else:
            self.net = net.to(self.device)
        # dataset
        self.train_set = data.DataLoader(dataset.MyData(train_path, img_size), batch_size=512, shuffle=True, num_workers=3)
        self.validation_set = data.DataLoader(dataset.MyData(validation_path, img_size), batch_size=512, shuffle=True)
        # optim
        self.opt = optim.SGD(self.net.parameters(), lr=0.0001)
        # loss
        self.loss_confi = nn.BCELoss()
        self.loss_offset = nn.MSELoss()
    def train(self):
        train_loss, validation_loss = [], []
        epoche = 1
        while True:
            print("[epoche] - {}:".format(epoche))
            self.net.train()
            for i, (x, confi, offset) in enumerate(self.train_set):
                x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
                c, o = self.net(x)
                c, o = c.view(-1, 1).to(self.device), o.view(-1, 4).to(self.device)
                # 剔除相应数据，置信度采用positive和negative，偏移量采用positive和part
                o, offset = o[confi.view(-1) != 0], offset[confi.view(-1) != 0]
                c, confi = c[confi.view(-1) != 2], confi[confi.view(-1) != 2]
                loss_c = self.loss_confi(c, confi)
                loss_o = self.loss_offset(o, offset)
                loss = loss_c+loss_o
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                train_loss.append(loss)
                print("process: {}/{}\nAlready finish {}!".format(i, len(self.train_set), round(i/len(self.train_set)*100, 2)))
            epoche += 1
            # 开启验证
            self.net.eval()
            _loss = self.validation()
            validation_loss.append(_loss)
            plt.clf()
            plt.plot(train_loss, "r", label="train_loss")
            plt.plot(validation_loss, "g", label="validation_loss")
            plt.pause(0.01)
            torch.save(self.net, self.save_path)
    def validation(self):
        for _x, _confi, _offset in self.validation_set:
            _c, _o = self.net(_x.cuda())
            _c, _o = _c.view(-1, 1).to(self.device), _o.view(-1, 4).to(self.device)
            _loss_c = self.loss_confi(_c, _confi.cuda())
            _loss_o = self.loss_offset(_o, _offset.cuda())
            _loss = _loss_c+_loss_o
            return _loss


            
                



            



        


