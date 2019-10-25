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
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


class MyTrain:
    def __init__(self):
        "Device"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        "Module"
        self.net = net.YoLoNet().to(self.device)
        if not os.path.exists("./params"):
            os.makedirs("./params")
        if not os.path.exists("./params/yolo.pkl"):
            print("Initing ... ...")
            self.net.apply(utils.weights_init)
        else:
            print("Loading ... ...")
            self.net.load_state_dict(torch.load("./params/yolo.pkl"))

        "Data"
        self.train = data.DataLoader(dataset.MyData(cfg.PATH), batch_size=cfg.BATCH, shuffle=True)

        "Optimize"
        self.opt = optim.Adam(self.net.parameters())

        "Loss"
        self.loss_confi = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_centre = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_box = nn.MSELoss(reduction='sum')
        self.loss_cls = nn.CrossEntropyLoss()

        "Show"
        self.draw = SummaryWriter("./run")

    def run(self, log: str, lower_loss=7.5):
        with open(log, "a+") as f:
            f.write("\n{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            for epoch in range(cfg.EPOCH):
                f.write("EPOCH - {}:\n".format(epoch))
                f.write("LOWER_LOSS - {}\n".format(lower_loss))
                for i, (label_13, label_26, label_52, img) in enumerate(self.train):
                    out_13, out_26, out_52 = self.net(img.to(self.device))
                    label_13, label_26, label_52 = label_13.to(self.device), label_26.to(self.device), label_52.to(self.device)
                    loss_13 = self.get_loss(out_13, label_13)
                    loss_26 = self.get_loss(out_26, label_26)
                    loss_52 = self.get_loss(out_52, label_52)
                    loss = loss_13+loss_26+loss_52
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    self.draw.add_scalar("LOSS", loss.item(), global_step=i)
                    print("EPOCH - {} - {}/{}".format(epoch, i, len(self.train)))
                if loss.item() < lower_loss:
                    lower_loss = loss.item()
                    torch.save(self.net.state_dict(), "./params/yolo.pkl")
                    f.write("SAVE COMPLETE! LOWER_LOSS - {}\n".format(lower_loss))
                f.flush()

    def get_loss(self, out, label, alpha=0.9):
        out = out.transpose(1, -1)
        out = out.reshape(out.size(0), out.size(1), out.size(2), 3, -1)
        mask_positive, mask_negative = label[..., 0] > 0, label[..., 0] == 0

        label_positive, label_negative = label[mask_positive], label[mask_negative]
        out_positive, out_negative = out[mask_positive], out[mask_negative]
        
        label_confi_positive, label_centre_positive, label_side_positive, label_cls_positive = label_positive[:, :1], label_positive[:, 1:3], label_positive[:, 3:5], label_positive[:, 5:]
        out_confi_positive, out_centre_positive, out_side_positive, out_cls_positive = out_positive[:, :1], out_positive[:, 1:3], out_positive[:, 3:5], out_positive[:, 5:]
        label_confi_negative, out_confi_negative = label_negative[:, :1], out_negative[:, :1]
        
        loss_confi = self.loss_confi(out_confi_positive, label_confi_positive)
        loss_centre = self.loss_centre(out_centre_positive, label_centre_positive)
        loss_box = self.loss_box(out_side_positive, label_side_positive)
        loss_cls = self.loss_cls(out_cls_positive, torch.argmax(label_cls_positive, dim=1))
        loss = alpha*(loss_confi+loss_centre+loss_box+loss_cls)+(1-alpha)*self.loss_confi(out_confi_negative, label_confi_negative)
        return loss


if __name__ == '__main__':
    log = "./log.txt"
    MyTrain().run(log)






        

        