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


class MyTrain:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Model
        self.trunk = net.MobileNet().to(self.device)
        # self.branch = net.ArcSoftmax(cfg.FEATURE_NUM, cfg.CLS_NUM, cfg.m, cfg.s).to(self.device)
        self.branch = net.CenterLoss(500, 2048).to(self.device)
        if not os.path.exists("./params"):
            os.mkdir("./params")
        if os.path.exists(cfg.TRUNK) and os.path.exists(cfg.BRANCH):
            print("Loading ... ...")
            self.trunk.load_state_dict(torch.load(cfg.TRUNK))
            self.branch.load_state_dict(torch.load(cfg.BRANCH))
        else:
            print("Initing ... ...")
            self.trunk.apply(utils.weights_init)
            self.branch.apply(utils.weights_init)
        # Data
        self.train = dataset.TRAIN
        self.test = dataset.TEST
        # Loss
        self.loss = nn.CrossEntropyLoss()
        # Optimize
        self.opt = optim.Adam(self.trunk.parameters())
        self.opt2 = optim.Adam(self.branch.parameters())

    def run(self, log: str, lower_loss=1000):
        with open(log, "a+") as f:
            # Configure Written
            f.write("\n{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write(">>> lamda: {}\n".format(0.5))
            # f.write(">>> m: {} >>> s: {}\n".format(cfg.m, cfg.s))
            # Train
            for epoch in range(cfg.EPOCH):
                f.write(">>> epoch: {}\n".format(epoch))
                f.write(">>> lower_loss: {}\n".format(lower_loss))
                self.trunk.train()
                a, b = [], []
                for i, (x, t) in enumerate(self.train):
                    x, t = x.to(self.device), t.to(self.device)
                    features, outputs = self.trunk(x)
                    loss_cross = self.loss(outputs, t)
                    loss_center = self.branch(features, t)
                    loss = loss_cross + 0.5 * loss_center
                    # Backward
                    self.opt.zero_grad()
                    self.opt2.zero_grad()
                    loss.backward()
                    self.opt2.step()
                    self.opt.step()
                    a.extend(outputs)
                    b.extend(t)
                    print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))
                a, b = torch.stack(a).cpu(), torch.stack(b).cpu()
                c = torch.mean((torch.argmax(a, dim=-1) == b).float())
                print(">>> accuracy: {}".format(c.item()))

                # Test
                with torch.no_grad():
                    self.trunk.eval()
                    output_list, target_list, loss_list = [], [], []
                    for x_, t_ in self.test:
                        x_, t_ = x_.to(self.device), t_.to(self.device)
                        features_, outputs_ = self.trunk(x_)
                        loss_cross_ = self.loss(outputs_, t_)
                        loss_center_ = self.branch(features_, t_)
                        loss_ = loss_cross_ + 0.5 * loss_center_

                        output_list.extend(outputs_)
                        target_list.extend(t_)
                        loss_list.append(loss_.item())

                    mean_loss = sum(loss_list) / len(loss_list)
                    f.write(">>> LOSS_MEAN: {}\n".format(mean_loss))
                    if mean_loss < lower_loss:
                        lower_loss = mean_loss
                        f.write(">>> SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                        torch.save(self.trunk.state_dict(), cfg.TRUNK)
                        torch.save(self.branch.state_dict(), cfg.BRANCH)

                    out, target = torch.stack(output_list).cpu(), torch.stack(target_list).cpu()
                    accuracy = torch.mean((torch.argmax(out, dim=-1) == target).float())
                    f.write(">>> Accuracy: {}%\n".format(accuracy * 100))

                f.flush()


if __name__ == '__main__':
    log = "./log.txt"
    MyTrain().run(log)
