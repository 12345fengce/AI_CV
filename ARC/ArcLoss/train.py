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
        # Optimize
        self.opt = optim.Adam(self.net.parameters())

    def run(self, log: str, lower_loss=100):
        with open(log, "a+") as f:
            # Configure Written
            f.write("\n{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(">>> m: {} >>> s: {}\n".format(cfg.m, cfg.s))
            # Train
            for epoch in range(cfg.EPOCH):
                f.write(">>> epoch: {}\n".format(epoch))
                f.write(">>> lower_loss: {}\n".format(lower_loss))
                self.net.train()
                for i, (x, t) in enumerate(self.train):
                    x, t = x.to(self.device), t.to(self.device)
                    features, _, loss = self.net(x, t)
                    print(loss.item())
                    # Backward
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))
                # Test
                with torch.no_grad():
                    self.net.eval()
                    output_list, target_list, coordinate_list, loss_list = [], [], [], []
                    for x_, t_ in self.test:
                        x_, t_ = x_.to(self.device), t_.to(self.device)
                        features_, outputs_, loss_ = self.net(x_, t_)

                        output_list.extend(outputs_)
                        target_list.extend(t_)
                        loss_list.append(loss_.item())
                        coordinate_list.extend(features_)

                    mean_loss = sum(loss_list)/len(loss_list)
                    if mean_loss < lower_loss:
                        lower_loss = mean_loss
                        f.write(">>> SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                        torch.save(self.net.state_dict(), cfg.MODEL)

                    out = torch.stack(output_list).cpu()
                    coordinate, target = torch.stack(coordinate_list).cpu(), torch.stack(target_list).cpu()
                    accuracy = torch.mean((torch.argmax(out, dim=-1) == target).float())
                    f.write(">>> Accuracy: {}%\n".format(accuracy*100))

                    plt.clf()
                    for num in range(10):
                        plt.scatter(coordinate[target == num, 0], coordinate[target == num, 1], c=cfg.COLOR[num], marker=".")
                    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], loc="upper right")
                    plt.title("[epoch] - {} >>> [Accuracy] - {:.2f}%".format(epoch, accuracy*100), loc="left")
                    plt.savefig("{}/pic{}.png".format(cfg.IMG, epoch))

                f.flush()


if __name__ == '__main__':
    log = "./log.txt"
    MyTrain().run(log)
