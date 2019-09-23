# -*-coding:utf-8-*-
import os
import torch
import dataset
import CenterLoss
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
        self.main_net = CenterLoss.MyNet().to(self.device)
        self.center_net = CenterLoss.CenterLoss(dataset.CLS, dataset.FEATURE).to(self.device)
        # 网络参数读取
        self.main_path = path+"/main.pt"
        self.center_path = path+"/center.pt"
        if os.path.exists(self.main_path):
            self.main_net.load_state_dict(torch.load(self.main_path))
            self.center_net.load_state_dict(torch.load(self.center_path))
        # 优化器
        self.main_optim = optim.Adam(self.main_net.parameters())
        self.center_optim = optim.SGD(self.center_net.parameters(), lr=0.5)
        self.lr_step = optim.lr_scheduler.StepLR(self.center_optim, step_size=50, gamma=0.9)
        # 损失
        self.loss_classify = nn.CrossEntropyLoss().to(self.device)
        self.loss_feature = CenterLoss.CenterLoss(dataset.CLS, dataset.FEATURE).to(self.device)
        self.write = SummaryWriter("./runs")

    def main(self, alpha=0.5):
        for epoche in range(2000):
            coordinate, target = [], []
            print("[epoche] - {}:".format(epoche))
            # 训练
            self.main_net.train()
            self.lr_step.step(epoche)
            for i, (x, t) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                features, outputs = self.main_net(x)
                loss_cls = self.loss_classify(outputs, t)
                loss_feat = self.loss_feature(features, t)
                loss = (1-alpha)*loss_cls+alpha*loss_feat
                self.main_optim.zero_grad()
                self.center_optim.zero_grad()
                loss.backward()
                self.main_optim.step()
                self.center_optim.step()
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
