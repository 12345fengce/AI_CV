# -*-coding:utf-8-*-
import os
import cfg
import net
import time
import utils
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class MyTrain:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        self.net = net.MyNet().to(self.device)
        if not os.path.exists(cfg.PATH) or not os.path.exists(cfg.IMG):
            os.makedirs(cfg.PATH)
            os.makedirs(cfg.IMG)
        if not os.path.exists(cfg.MODEL):
            print("Initing ... ...")
            self.net.apply(utils.weights_init)
        else:
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(cfg.MODEL))
        # Data
        self.train = dataset.TRAIN
        self.test = dataset.TEST
        # Loss
        self.centerloss = net.CenterLoss(cfg.CLS_NUM, cfg.FEATURE_NUM).to(self.device)
        if os.path.exists(cfg.CLOSS):
            self.centerloss.load_state_dict(torch.load(cfg.CLOSS))
        self.clsloss = nn.CrossEntropyLoss()
        # Optimize
        self.opt = optim.Adam(self.net.parameters())
        self.opt_centerloss = optim.Adam(self.centerloss.parameters())

    def run(self, log: str, lower_loss=1.0):
        with open(log, "a+") as f:
            # Configure Written
            f.write("\n{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(">>> lambda: {}\n".format(cfg.LAMBDA))
            # Train
            for epoch in range(101, cfg.EPOCH):
                f.write(">>> epoch: {}\n".format(epoch))
                f.write(">>> lower_loss: {}\n".format(lower_loss))
                self.net.train()
                for i, (x, t) in enumerate(self.train):
                    x, t = x.to(self.device), t.to(self.device)
                    features, outputs = self.net(x)
                    loss_center = self.centerloss(features, t)
                    loss_cls = self.clsloss(outputs, t)
                    loss = loss_cls+cfg.LAMBDA*loss_center

                    # Backward
                    self.opt.zero_grad()
                    self.opt_centerloss.zero_grad()
                    loss.backward()
                    self.opt.step()
                    self.opt_centerloss.step()
                    print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))

                # Test
                with torch.no_grad():
                    self.net.eval()
                    out, target, coordinate, Loss = [], [], [], []
                    for x_, t_ in self.test:
                        x_, t_ = x_.to(self.device), t_.to(self.device)
                        features_, outputs_ = self.net(x_)
                        loss_center_ = self.centerloss(features_, t_)
                        loss_cls_ = self.clsloss(outputs_, t_)
                        loss_ = loss_cls_ + cfg.LAMBDA * loss_center_

                        out.extend(torch.softmax(outputs_, dim=-1))
                        target.extend(t_)
                        Loss.append(loss_.item())
                        coordinate.extend(features_)

                    mean_loss = sum(Loss)/len(Loss)
                    if mean_loss < lower_loss:
                        lower_loss = mean_loss
                        f.write(">>> SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                        torch.save(self.net.state_dict(), cfg.MODEL)
                        torch.save(self.centerloss.state_dict(), cfg.CLOSS)

                    out = torch.stack(out).cpu()
                    coordinate, target = torch.stack(coordinate).cpu(), torch.stack(target).cpu()
                    accuracy = torch.mean((torch.argmax(out, dim=-1) == target).float())
                    f.write(">>> Accuracy: {}%\n".format(accuracy*100))

                    plt.clf()
                    for num in range(10):
                        plt.scatter(coordinate[target == num, 0], coordinate[target == num, 1], c=cfg.COLOR[num], marker=".")
                    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
                    center = self.centerloss.center.data.cpu().numpy()
                    plt.scatter(center[:, 0], center[:, 1], c=cfg.COLOR[10])
                    plt.title("[epoch] - {} >>> [Accuracy] - {:.2f}%".format(epoch, accuracy*100), loc="left")
                    plt.savefig("{}/pic{}.png".format(cfg.IMG, epoch))

                f.flush()


if __name__ == '__main__':
    log = "./log/clog.txt"
    MyTrain().run(log)
