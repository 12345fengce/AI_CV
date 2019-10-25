# -*-coding:utf-8-*-
import os
import net
import cfg
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

        # Module
        self.net = net.MobileNet(len(cfg.NAME)).to(self.device)
        dir = "./params"
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.params = dir+"/10_20.pkl"
        if not os.path.exists(self.params):
            print("Initing ... ...")
            self.net.apply(utils.weights_init)
        else:
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(self.params))

        # Data
        self.train = dataset.TRAIN

        "Optimize"
        self.opt = optim.Adam(self.net.parameters())

        "Loss"
        self.loss_confi = nn.MSELoss(reduction='sum')
        self.loss_centre = nn.MSELoss(reduction='sum')
        self.loss_box = nn.MSELoss(reduction='sum')
        self.loss_cls = nn.CrossEntropyLoss()

    def run(self, lower_loss=5):
        for epoch in range(cfg.EPOCH):
            self.net.train()
            for i, (img, label) in enumerate(self.train):
                img, label = img.to(self.device), label.to(self.device)
                outputs = self.net(img)
                loss = self.get_loss(outputs, label)
                print("EPOCH - {} - {}/{} >>> loss: {}".format(epoch, i, len(self.train), loss.item()))

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if loss.item() < lower_loss:
                lower_loss = loss.item()
                torch.save(self.net.state_dict(), self.params)

    def get_loss(self, out, label, alpha=0.9):
        out = out.transpose(1, -1)
        out = out.reshape(shape=(out.size(0), out.size(1), out.size(2), 3, -1))
        mask_positive, mask_negative = label[..., 0] > 0, label[..., 0] == 0

        label_positive, label_negative = label[mask_positive], label[mask_negative]
        out_positive, out_negative = out[mask_positive], out[mask_negative]
        
        label_confi_positive, label_centre_positive, label_side_positive, label_cls_positive = label_positive[:, :1], label_positive[:, 1:3], label_positive[:, 3:5], label_positive[:, 5].long()
        out_confi_positive, out_centre_positive, out_side_positive, out_cls_positive = out_positive[:, :1], out_positive[:, 1:3], out_positive[:, 3:5], out_positive[:, 5:]
        label_confi_negative, out_confi_negative = label_negative[:, :1], out_negative[:, :1]
        
        loss_confi = self.loss_confi(out_confi_positive, label_confi_positive)
        loss_centre = self.loss_centre(out_centre_positive, label_centre_positive)
        loss_box = self.loss_box(out_side_positive, label_side_positive)
        loss_cls = self.loss_cls(out_cls_positive, label_cls_positive)
        loss = alpha*(loss_confi+loss_centre+loss_box+loss_cls)+(1-alpha)*self.loss_confi(out_confi_negative, label_confi_negative)
        return loss


if __name__ == '__main__':
    MyTrain().run()






        

        