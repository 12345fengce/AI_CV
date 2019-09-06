# -*-coding:utf-8-*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, mode: str, batch_size: int):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net
        self.mode = mode
        if mode == "P" or mode == "p":
            self.size = 12
            self.threshold = 0.9
            self.net = net.PNet().to(self.device)
        elif mode == "R" or mode == "r":
            self.size = 24
            self.threshold = 0.99
            self.net = net.RNet().to(self.device)
        elif mode == "O" or mode == "o":
            self.size = 48
            self.threshold = 0.999
            self.net = net.ONet().to(self.device)
        if len(os.listdir("./params")) > 0:
            print("MODE: {} >>> Loading ... ...".format(mode))
            self.net.load_state_dict(torch.load("./params/{}net.pkl".format(mode.lower())))
        # dataloader
        self.train = data.DataLoader(dataset.choice(mode.lower()), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.test = data.DataLoader(dataset.choice("{}v".format(mode.lower())), batch_size=512, shuffle=True, drop_last=True)
        # optimize
        self.optimize = optim.SGD(self.net.parameters(), lr=3e-4, momentum=0.9, weight_decay=1e-5)
        # loss
        self.loss_confi = nn.BCELoss()
        self.loss_offset = nn.MSELoss()
        # show
        self.summarywriter = SummaryWriter(log_dir="./runs/{}_runs".format(mode.lower()))

    def main(self):
        for epoche in range(33333):
            # train
            self.net.train()
            for i, (x, confi, offset) in enumerate(self.train):
                x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
                cout, oout = self.net(x)
                # filter
                cout, oout = cout.view(x.size(0), -1), oout.view(x.size(0), -1)
                accurary, recall = self.critic(cout, oout, confi, offset)
                oout, offset = oout[confi.view(-1) != 0], offset[confi.view(-1) != 0]
                cout, confi = cout[confi.view(-1) != 2], confi[confi.view(-1) != 2]
                # loss
                closs = self.loss_confi(cout, confi)
                oloss = self.loss_offset(oout, offset)
                regular = net.Regular(self.net, weight_decay=1e-5)
                if self.mode == "p" or self.mode == "P":
                    loss = 2*closs+oloss+regular.regular_loss()
                elif self.mode == "r" or self.mode == "R":
                    loss = closs+oloss+regular.regular_loss()
                elif self.mode == "o" or self.mode == "O":
                    loss = closs+2*oloss+regular.regular_loss()
                # backward
                self.optimize.zero_grad()
                loss.backward()
                self.optimize.step()
                # show
                self.summarywriter.add_scalar("LOSS-TRAIN", loss.item(), global_step=i)
                print("Proccessing: {}/{}".format(i, len(self.train)))
                print("$ 训练集：[epoche] - {}  Accuracy: {:.2f}%  Recall: {:.2f}%"
                      .format(epoche, accurary.item()*100, recall.item()*100))
            # test
            self.net.eval()
            with torch.no_grad():
                for _x, _confi, _offset in self.test:
                    _x, _confi, _offset = _x.to(self.device), _confi.to(self.device), _offset.to(self.device)
                    _cout, _oout = self.net(_x)
                    # filter
                    _cout, _oout = _cout.view(_x.size(0), -1), _oout.view(_x.size(0), -1)
                    _accurary, _recall = self.critic(_cout, _oout, _confi, _offset)
                    _oout, _offset = _oout[_confi.view(-1) != 0], _offset[_confi.view(-1) != 0]
                    _cout, _confi = _cout[_confi.view(-1) != 2], _confi[_confi.view(-1) != 2]
                    # loss
                    _closs = self.loss_confi(_cout, _confi)
                    _oloss = self.loss_offset(_oout, _offset)
                    _loss = _closs+_oloss
                    self.summarywriter.add_scalar("LOSS-TEST", _loss.item(), global_step=epoche)
                    print("$ 训练集：[epoche] - {}  Accuracy: {:.2f}%  Recall: {:.2f}% "
                          .format(epoche, _accurary.item() * 100, _recall.item() * 100))
            torch.save(self.net.state_dict(), "./params/{}net.pkl")

    def critic(self, cout, oout, confi, offset):
        TP = torch.sum(cout[confi.view(-1) != 0] >= self.threshold)
        FP = torch.sum(cout[confi.view(-1) == 0] >= self.threshold)
        TN = torch.sum(cout[confi.view(-1) == 0] < 0.1)
        FN = torch.sum(cout[confi.view(-1) != 0] < 0.1)
        accurary = (TP+TN).float()/(TP+FP+TN+FN).float()
        recall = TP.float()/(TP+FN).float()

        return accurary, recall






            
                



            



        


