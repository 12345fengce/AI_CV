# -*-coding:utf-8-*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class MyTrainer:
    def __init__(self, save_path, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        if os.path.exists(save_path):
            self.net = torch.load(save_path)
        else:
            self.net = net.YoLoNet().to(self.device)
        self.net.train()
        self.train = data.DataLoader(dataset.MyData(data_path), batch_size=3, shuffle=True)
        self.opt = optim.Adam(self.net.parameters())
        self.loss = nn.MSELoss()
    def optimize(self):
        count = 1
        while True:
            for i, (label_13, label_26, label_52, img) in enumerate(self.train):
                out_13, out_26, out_52 = self.net(img.to(self.device))
                label_13, label_26, label_52 = label_13.to(self.device), label_26.to(self.device), label_52.to(self.device)
                loss_13 = self.transform(out_13, label_13)
                loss_26 = self.transform(out_26, label_26)
                loss_52 = self.transform(out_52, label_52)
                loss = loss_13+loss_26+loss_52
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            count += 1
            if count%10 == 0:
                print("[epoche] - {} - loss_13:{} + loss_26:{} + loss_52:{} = Loss:{}".format(count, loss_13, loss_26, loss_52, loss))
                torch.save(self.net, self.save_path)
    def transform(self, out, label, α=0.9):
        """out: (N, C, H, W)
            label: (N, W, H, 3, 5+cls_num)
            [iou, x_offset, y_offset, w_offset, h_offset, cls:one_hot]"""
        out = out.transpose(1, -1)
        out = out.reshape(out.size(0), out.size(1), out.size(2), 3, -1)
        mask_positive, mask_negative = label[..., 0] > 0, label[..., 0] == 0
        loss_positive = self.loss(out[mask_positive], label[mask_positive])
        loss_negative = self.loss(out[mask_negative], label[mask_negative])
        loss = loss_positive*α+loss_negative*(1-α)
        return loss







        

        