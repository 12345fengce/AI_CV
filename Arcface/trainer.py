# -*-coding:utf-8-*-
import os
import net
import torch
import dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, path, alpha, cls=10, features=2):
        self.path = path
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(path):
            print("loading... ...")
            self.net = torch.load(path)
        else:
            print("creating... ...")
            self.net = net.MyNet().to(self.device)
        self.train = data.DataLoader(dataset.train_data, batch_size=512, shuffle=True)
        self.test = data.DataLoader(dataset.test_data, batch_size=512, shuffle=False)
        self.opt = optim.Adam(self.net.parameters())
        self.loss_classify = nn.CrossEntropyLoss().to(self.device)
        self.loss_feature = net.CenterLoss(cls, features).to(self.device)
    def main(self):
        Accuracy = []
        epoche = 1
        plt.ion()
        while True:
            print("[epoche] - {}:".format(epoche))
            self.net.train()
            for i, (data, label) in enumerate(self.train):
                data, label = data.to(self.device), label.to(self.device)
                feature, output = self.net(data)
                loss_feat = self.loss_feature(feature, label)
                loss_cls = self.loss_classify(output, label)
                loss = loss_cls+self.alpha*loss_feat
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                coordinate = feature.data
                color = [dataset.color[index] for index in label.data.cpu().numpy()]
            plt.clf()
            plt.title("[epoche] - {}".format(epoche), loc='left')
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
            plt.scatter(coordinate[:, 0].cpu(), coordinate[:, 1].cpu(), c=color)
            plt.pause(0.01)

            self.net.eval()
            _loss_feat, _loss_cls, _loss, accuracy = self.valudation()
            Accuracy.append(accuracy.item())
            print("MODE - TRAIN\nLoss_classify:{} + Loss_feature:{} = Loss:{}".format(loss_cls, loss_feat, loss))
            print("MODE - VALUDATION\nLoss_classify:{} + Loss_feature:{} = Loss:{}".format(_loss_cls, _loss_feat, _loss))
            if epoche % 10 == 0:
                torch.save(self.net, self.path)
                plt.savefig(self.path.replace(".pt", ".jpg"))
            epoche += 1
    def valudation(self):
        for data, label in self.test:
            data, label = data.to(self.device), label.to(self.device)
            feature, output = self.net(data)
            _output = nn.functional.softmax(output, dim=1)
            accuracy = torch.mean(torch.argmax(_output, dim=1) == label, dtype=torch.float)
            loss_feat = self.loss_feature(feature, label)
            loss_cls = self.loss_classify(output, label)
            loss = loss_cls+loss_feat
            return loss_feat, loss_cls, loss, accuracy





            