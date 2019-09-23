# -*-coding:utf-8-*-
import os
import net
import sys
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
from lookahead import Lookahead
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, mode: str, batch_size: int):
        "Device Config"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        "Model Config"
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
        if len(os.listdir("./params")) > 3:
            print("MODE: {} >>> Loading ... ...".format(mode))
            self.net.load_state_dict(torch.load("./params/{}net.pkl".format(mode.lower())))

        "Dataloader Config"
        self.train = data.DataLoader(dataset.choice(mode.lower()), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.test = data.DataLoader(dataset.choice("{}v".format(mode.lower())), batch_size=512, shuffle=True, drop_last=True)

        "Optim Config"
        optimize = optim.SGD(self.net.parameters(), lr=3e-4, momentum=0.9)
        self.lookahead = Lookahead(optimize, k=5, alpha=0.5)
        # self.lr = optim.lr_scheduler.CosineAnnealingLR(self.lookahead, T_max=1, eta_min=1e-5, last_epoch=-1)

        "Loss Config"
        self.loss_confi = nn.BCELoss()
        self.loss_resolve = nn.BCELoss(reduction="none")
        self.loss_offset = nn.MSELoss()

        "Show Config"
        self.summarywriter = SummaryWriter(log_dir="./runs/{}_runs".format(mode.lower()))

    def main(self):
        for epoche in range(33333):
            "MODE：TRAIN"
            self.net.train()
            for i, (x, confi, offset) in enumerate(self.train):
                x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
                cout, oout = self.net(x)

                "Critic"
                cout, oout = cout.view(x.size(0), -1), oout.view(x.size(0), -1)
                accurary, recall = self.critic(cout, confi)

                "Filter"
                confi_mask, offset_mask = confi.view(-1) != 2, confi.view(-1) != 0
                oout, offset = oout[offset_mask], offset[offset_mask]
                cout, confi = cout[confi_mask], confi[confi_mask]

                "Loss"
                # closs = self.loss_confi(cout, confi)
                oloss = self.loss_offset(oout, offset)
                hloss = self.hardsample(x, cout, confi, confi_mask)
                # regular = net.Regular(self.net, weight_decay=0)
                if self.mode == "p" or self.mode == "P":
                    loss = 2*closs+oloss+regular.regular_loss()
                elif self.mode == "r" or self.mode == "R":
                    loss = closs+oloss+regular.regular_loss()
                elif self.mode == "o" or self.mode == "O":
                    loss = hloss+2*oloss

                "BackWard"
                self.lookahead.zero_grad()
                loss.backward()
                self.lookahead.step()

                "Show"
                self.summarywriter.add_scalar("LOSS-TRAIN", loss.item(), global_step=i)
                print("Proccessing: {}/{}".format(i, len(self.train)))
                print("$ 训练集：[epoche] - {}  Accuracy: {:.2f}%  Recall: {:.2f}%"
                      .format(epoche, accurary.item()*100, recall.item()*100))
            # self.lr.step(epoche)

            "MODE: VERIFY"
            self.net.eval()
            with torch.no_grad():
                for _x, _confi, _offset in self.test:
                    _x, _confi, _offset = _x.to(self.device), _confi.to(self.device), _offset.to(self.device)
                    _cout, _oout = self.net(_x)

                    "Critic"
                    _cout, _oout = _cout.view(_x.size(0), -1), _oout.view(_x.size(0), -1)
                    _accurary, _recall = self.critic(_cout, _confi)

                    "Filter"
                    _oout, _offset = _oout[_confi.view(-1) != 0], _offset[_confi.view(-1) != 0]
                    _cout, _confi = _cout[_confi.view(-1) != 2], _confi[_confi.view(-1) != 2]

                    "Loss"
                    _closs = self.loss_confi(_cout, _confi)
                    _oloss = self.loss_offset(_oout, _offset)
                    _loss = _closs+_oloss

                    "Show"
                    self.summarywriter.add_scalar("LOSS-TEST", _loss.item(), global_step=epoche)
                    print("$ 训练集：[epoche] - {}  Accuracy: {:.2f}%  Recall: {:.2f}% "
                          .format(epoche, _accurary.item() * 100, _recall.item() * 100))

            "MODE: SAVE"
            torch.save(self.net.state_dict(), "./params/{}net.pkl".format(self.mode.lower()))

    def critic(self, cout, confi):
        "Accurary、Recall"
        TP = torch.sum(cout[confi.view(-1) != 0] > self.threshold)
        FP = torch.sum(cout[confi.view(-1) == 0] > 0.1)
        TN = torch.sum(cout[confi.view(-1) == 0] < 0.1)
        FN = torch.sum(cout[confi.view(-1) != 0] < self.threshold)
        accurary = (TP+TN).float()/(TP+FP+TN+FN).float()
        recall = TP.float()/(TP+FN).float()
        return accurary, recall

    def hardsample(self, x, out, label, idx):
        "Hard Sample"
        resolve_loss = self.loss_resolve(out, label)
        _, hard_index = torch.sort(resolve_loss, dim=0)
        hard_mask = hard_index[int(hard_index.size(0) * 0.3):].squeeze()
        hard_x = x[idx][hard_mask]
        hard_label = label[hard_mask]
        hard_out, _ = self.net(hard_x)
        hard_loss = self.loss_confi(hard_out, hard_label)
        return hard_loss






            
                



            



        


