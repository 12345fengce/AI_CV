# -*- coding:utf-8 -*-
import os
import net
import cfg
import time
import torch
import dataset
import functions
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class MyTrain:
    def __init__(self, mode: str):
        "Device"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        "Net"
        self.name = mode.lower()+"net"
        if mode == "P" or mode == "p":
            self.net = net.PNet().to(self.device)
            self.train = dataset.P_TRAIN
            self.test = dataset.P_TEST
        elif mode == "R" or mode == "r":
            self.net = net.RNet().to(self.device)
            self.train = dataset.R_TRAIN
            self.test = dataset.R_TEST
        elif mode == "O" or mode == "o":
            self.net = net.ONet().to(self.device)
            self.train = dataset.O_TRAIN
            self.test = dataset.O_TEST

        if os.path.exists("./params/{}.pkl".format(self.name)):
            print("MODE: {} >>> Loading ... ...".format(mode.upper()))
            self.net.load_state_dict(torch.load("./params/{}.pkl".format(self.name)))
        else:
            print("MODE: {} >>> Initing ... ...".format(mode.upper()))
            self.net.apply(functions.weights_init)

        "Optimize"
        self.opt = optim.Adam(self.net.parameters())

        "Loss"
        self.Closs = nn.BCEWithLogitsLoss()
        self.Hloss = nn.BCEWithLogitsLoss(reduction='none')
        self.Oloss = nn.MSELoss()
        self.LMloss = nn.MSELoss()

        "Tensorboard"
        self.draw = SummaryWriter(log_dir="./runs/{}_runs".format(mode.lower()))

    def run(self, log: str):
        with open(log, "w+") as f:
            f.write("{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write("Configure:\nBatch: {}\nLoss weights: confi_loss: offset_loss: landmark_loss = {}: {}: {}\n"
                    .format(cfg.BATCH[self.name], cfg.LOSS[self.name][0], cfg.LOSS[self.name][1], cfg.LOSS[self.name][2]))

            for epoch in range(cfg.EPOCH):
                f.write("Epoche - {}\n".format(epoch))
                "Train"
                self.net.train()
                for i, (x, confi, offset, landmark) in enumerate(self.train):
                    x, confi, offset, landmark = x.to(self.device), confi.to(self.device), \
                                                 offset.to(self.device), landmark.to(self.device)
                    Cout, Oout, LMout = self.net(x)
                    Cout, Oout, LMout = Cout.view(x.size(0), -1), Oout.view(x.size(0), -1), LMout.view(x.size(0), -1)

                    "Filter"
                    Cmask, Omask, LMmask = confi.view(-1) != 2, confi.view(-1) != 0, confi.view(-1) == 1
                    Cout, confi = Cout[Cmask], confi[Cmask]
                    Oout, offset = Oout[Omask], offset[Omask]
                    LMout, landmark = LMout[LMmask], landmark[LMmask]

                    "Loss"
                    Hloss = self.hardsample(x[Cmask], Cout, confi)
                    Oloss = self.Oloss(Oout, offset)
                    LMloss = self.LMloss(LMout, landmark)
                    loss = Hloss*cfg.LOSS[self.name][0]+Oloss*cfg.LOSS[self.name][1]+LMloss*cfg.LOSS[self.name][2]

                    "BackWard"
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    "Show"
                    self.draw.add_scalar("LOSS-TRAIN", loss.item(), global_step=i)
                    print("EPOCHE: {}\nRUNNING ! ! ! SCHEDULE: {:.2f}%".format(epoch, i/len(self.train)*100))
                "MODE: VERIFY"
                self.net.eval()
                with torch.no_grad():
                    out, label, Loss = [], [], []
                    for _x, _confi, _offset, _landmark in self.test:
                        _x, _confi, _offset, _landmark = _x.to(self.device), _confi.to(self.device),\
                                                         _offset.to(self.device), _landmark.to(self.device)
                        _Cout, _Oout, _LMout = self.net(_x)

                        "Critic"
                        _Cout, _Oout, _LMout = _Cout.view(_x.size(0), -1), _Oout.view(_x.size(0), -1), _LMout.view(_x.size(0), -1)
                        _Omask, _LMmask = _confi.view(-1) != 0, _confi.view(-1) == 1
                        _Oout, _offset = _Oout[_Omask], _offset[_Omask]
                        _LMout, _landmark = _LMout[_LMmask], _landmark[_LMmask]
                        _loss = self.Oloss(_Oout, _offset)+self.LMloss(_LMout, _landmark)

                        out.append(_Cout)
                        label.append(_confi)
                        Loss.append(_loss.item())

                    out, label = torch.cat(out).to(self.device), torch.cat(label).to(self.device)
                    accuracy, recall = self.critic(out, label)
                    f.write("TEST  Accuracy: {}  Recall:{}\nLower error: {}\n".format(accuracy.item(), recall.item(),
                                                                                      cfg.LOWER_ERROR[self.name]))
                    error = sum(Loss)/len(Loss)

                    "Show"
                    self.draw.add_scalar("LOSS-TEST", error, global_step=epoch)

                "MODE: SAVE"
                if error < cfg.LOWER_ERROR[self.name]:
                    cfg.LOWER_ERROR[self.name] = error
                    torch.save(self.net.state_dict(), "./params/{}.pkl".format(self.name))
                    f.write("Lower_error: {} SAVE COMPLETE!\n".format(cfg.LOWER_ERROR[self.name]))
                f.flush()

    def critic(self, cout, confi):
        "Accurary、Recall"
        TP = torch.sum(cout[confi.view(-1) != 0] > cfg.CONFI[self.name])
        FP = torch.sum(cout[confi.view(-1) == 0] > 0.1)
        TN = torch.sum(cout[confi.view(-1) == 0] < 0.1)
        FN = torch.sum(cout[confi.view(-1) != 0] < cfg.CONFI[self.name])
        accurary = (TP + TN).float() / (TP + FP + TN + FN).float()
        recall = TP.float() / (TP + FN).float()
        return accurary, recall

    def hardsample(self, x, out, label):
        "Hard Sample"
        Rloss = self.Hloss(out, label)
        _, index = torch.sort(Rloss, dim=0)
        mask = index[int(index.size(0) * 0.3):].squeeze()
        Hx, Hlabel = x[mask], label[mask]
        Hout, *_ = self.net(Hx)
        Hloss = self.Closs(Hout.view(-1, 1), Hlabel)
        return Hloss


if __name__ == '__main__':
    log = "./Plog.txt"
    mytrain = MyTrain("P")
    mytrain.run(log)