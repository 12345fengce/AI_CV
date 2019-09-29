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
        self.net = net.MyNet02().to(self.device)
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
        self.arcsoftmax = net.ArcFace(cfg.CLS_NUM, cfg.FEATURE_NUM).to(self.device)
        if os.path.exists(cfg.AFACE):
            self.arcsoftmax.load_state_dict(torch.load(cfg.AFACE))
        self.loss = nn.NLLLoss()
        # Optimize
        self.opt = optim.Adam(self.net.parameters())
        self.opt2 = optim.SGD(self.centerloss.parameters(), lr=0.1)
        self.lr_step = optim.lr_scheduler.StepLR(self.opt2, step_size=50, gamma=0.3)

    def run(self, log: str, lower_loss=6):
        with open(log, "a+") as f:
            # Configure Written
            m, s = cfg.ARC[0]
            f.write("\n{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(">>> m: {} >>> s: {}\n".format(m, s))
            # Train
            for epoch in range(cfg.EPOCH):
                f.write(">>> epoch: {}\n".format(epoch))
                f.write(">>> lower_loss: {}\n".format(lower_loss))
                self.net.train()
                for i, (x, t) in enumerate(self.train):
                    x, t = x.to(self.device), t.to(self.device)
                    features = self.net(x)
                    outputs = self.arcsoftmax(features, m, s)
                    loss = self.loss(torch.log(outputs), t)
                    # Backward
                    self.opt.zero_grad()
                    self.opt2.zero_grad()
                    loss.backward()
                    self.opt2.step()
                    self.opt.step()
                    print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))
                    # self.write.add_scalar("train >>> loss: ", loss.item(), i)
                # Test
                with torch.no_grad():
                    self.net.eval()
                    out, target, coordinate, Loss = [], [], [], []
                    for x_, t_ in self.test:
                        x_, t_ = x_.to(self.device), t_.to(self.device)
                        features_ = self.net(x_)
                        outputs_ = self.arcsoftmax(features_, m, s)
                        loss_ = self.loss(torch.log(outputs_), t_)

                        out.extend(torch.softmax(outputs_, dim=-1))
                        target.extend(t_)
                        Loss.append(loss_.item())
                        coordinate.extend(features_)

                    mean_loss = sum(Loss)/len(Loss)
                    # self.write.add_scalar("test >>> loss: ", mean_loss, epoch)
                    if mean_loss < lower_loss:
                        lower_loss = mean_loss
                        f.write(">>> SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                        torch.save(self.net.state_dict(), cfg.MODEL)
                        torch.save(self.arcsoftmax.state_dict(), cfg.AFACE)

                    out = torch.stack(out).cpu()
                    coordinate, target = torch.stack(coordinate).cpu(), torch.stack(target).cpu()
                    accuracy = torch.mean((torch.argmax(out, dim=-1) == target).float())
                    f.write(">>> Accuracy: {}%\n".format(accuracy*100))

                    plt.clf()
                    for num in range(10):
                        plt.scatter(coordinate[target == num, 0], coordinate[target == num, 1], c=cfg.COLOR[num], marker=".")
                    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
                    plt.title("[epoch] - {}".format(epoch), loc="left")
                    plt.savefig("{}/pic{}.png".format(cfg.IMG, epoch))

                f.flush()


if __name__ == '__main__':
    log = "./log/alog.txt"
    MyTrain().run(log)
