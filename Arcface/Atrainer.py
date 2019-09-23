# -*-coding:utf-8-*-
import os
import torch
import dataset
import ArcLoss
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, path):
        # 数据集
        self.train = data.DataLoader(dataset.train_data, batch_size=8, shuffle=True)
        self.test = data.DataLoader(dataset.test_data, batch_size=128, shuffle=True)
        # 网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.main_net = ArcLoss.MyNet().to(self.device)
        self.arcloss = ArcLoss.ArcFace(dataset.FEATURE, dataset.CLS).to(self.device)
        # 网络参数读取
        self.main_path = path+"/Amain.pt"
        self.arcloss = path+"/ArcLoss.pt"
        if os.path.exists(self.main_path):
            self.main_net.load_state_dict(torch.load(self.main_path))
            self.arcloss.load_state_dict(torch.load(self.arcloss))
        # 优化器
        self.main_optim = optim.Adam(self.main_net.parameters())
        self.arc_optim = optim.SGD(self.arcloss.parameters(), lr=0.5)
        self.lr_step = optim.lr_scheduler.StepLR(self.arc_optim, step_size=50, gamma=0.9)
        self.write = SummaryWriter("./runs")

    def main(self, alpha=0.1):
        for epoche in range(200):
            coordinate, target = [], []
            print("[epoche] - {}:".format(epoche))
            # 训练
            self.main_net.train()
            self.lr_step.step(epoche)
            for i, (x, t) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                features = self.main_net(x)
                loss = self.arcloss(features)
                self.main_optim.zero_grad()
                self.arc_optim.zero_grad()
                loss.backward()
                self.main_optim.step()
                self.arc_optim.step()
                coordinate.append(features)
                target.append(t)
                print("MODE - TRAIN\nLoss_cls:{} + Loss_feat:{} = Loss:{}".format(loss_cls, loss_feat, loss))
                self.write.add_scalar("train >>> loss: ", loss.item(), i)
            coordinates = torch.cat(coordinate, 0).data.cpu()
            targets = torch.cat(target, 0).data.cpu()
            plt.clf()
            for num in range(10):
                plt.scatter(coordinates[targets == num, 0], coordinates[targets == num, 1], c=dataset.COLOR[num])
            plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
            plt.title("[epoche] - {}".format(epoche), loc="left")
            plt.pause(0.01)
            plt.savefig("./img/center{}.png".format(epoche))
            # 验证
            self.main_net.eval()
            _loss_feat, _loss_cls, _accuracy = self.verify()
            print("test*****************************************************************************************")
            self.write.add_scalar("test >>> loss: ", 0.9*_loss_cls+0.1*loss_feat, epoche)
            print("Accuracy:{}%\nLoss_classify:{} && Loss_feature:{}"
                  .format(_loss_cls, alpha*_loss_feat, _accuracy*100))
            if epoche % 10 == 0:
                torch.save(self.main_net.state_dict(), self.main_path)
                torch.save(self.center_net.state_dict(), self.center_path)
            epoche += 1

    def verify(self):
        for img, label in self.test:
            img, label = img.to(self.device), label.to(self.device)
            feature, output = self.main_net(img)
            _output = nn.functional.softmax(output, dim=1)
            accuracy = torch.mean(torch.argmax(_output, dim=1) == label, dtype=torch.float)
            loss_feat = self.loss_feature(feature, label)
            loss_cls = self.loss_classify(output, label)
            return loss_feat, loss_cls, accuracy


if __name__ == '__main__':
    mytrain = Trainer("./params")
    mytrain.main()
