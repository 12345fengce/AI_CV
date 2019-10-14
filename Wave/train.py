# -*- coding:utf-8 -*-
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
        self.net = net.MyNet().to(self.device)
        if not os.path.exists(cfg.PARAMS):
            if not os.path.exists("./params"):
                os.mkdir("./params")
            print("Initing ... ...")
            self.net.apply(utils.weights_init)
        else:
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(cfg.PARAMS))
        # Data
        self.train = dataset.TRAIN
        self.test = dataset.TEST
        # Loss
        self.loss = nn.CrossEntropyLoss()
        # Optimize
        self.opt = optim.Adam(self.net.parameters())

    def run(self, log: str, lower_loss=1):
        with open(log, "a+") as f:
            # Configure Written
            f.write("{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            # Train
            for epoch in range(cfg.EPOCH):
                f.write(">>> epoch: {}\n".format(epoch))
                self.net.train()
                for i, (x, t) in enumerate(self.train):
                    x, t = x.to(self.device), t.to(self.device)
                    output = self.net(x)
                    loss = self.loss(output, t)
                    # Backward
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    print("epoch >>> {} >>> {}/{}".format(epoch, i, len(self.train)))

                self.net.eval()
                with torch.no_grad():
                    loss_item, accuracy_item = [], []
                    for x_, t_ in self.test:
                        x_, t_ = x_.to(self.device), t_.to(self.device)
                        output_ = self.net(x_)
                        loss_ = self.loss(output_, t_)
                        loss_item.append(loss_.item())
                        accuracy = torch.mean((torch.argmax(output_, dim=-1) == t_).float())
                        accuracy_item.append(accuracy)
                    loss_mean = sum(loss_item)/len(loss_item)
                    accuracy_mean = sum(accuracy_item)/len(accuracy_item)
                    f.write(">>> Test Accuracy: {}\n".format(accuracy_mean))
                    # Save
                    if loss_mean < lower_loss:
                        lower_loss = loss_mean
                        f.write(">>> SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                        torch.save(self.net.state_dict(), cfg.PARAMS)
                f.flush()


if __name__ == '__main__':
    log = "./log.txt"
    MyTrain().run(log)

