# -*-coding:utf-8-*-
import os
import net
import torch
import random
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class Trainer:
    def __init__(self, net, train_path, validation_path, save_path, img_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.img_size = img_size
        if os.path.exists(save_path):
            self.net = torch.load(save_path)
        else:
            self.net = net.to(self.device)
        self.train_set = data.DataLoader(dataset.MyData(train_path, img_size), batch_size=512, shuffle=True, num_workers=15)
        self.validation_set = data.DataLoader(dataset.MyData(validation_path, img_size), batch_size=512, shuffle=True, num_workers=15)
        self.opt = optim.SGD(self.net.parameters(), lr=0.001)
        self.loss_confi = nn.BCELoss()
        self.loss_offset = nn.MSELoss()

    def train(self):
        epoche = 1
        while True:
            print("[epoche] - {}:".format(epoche))
            self.net.train()
            out_correct, label_correct = 0, 0
            for i, (x, confi, offset) in enumerate(self.train_set):
                x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
                c, o = self.net(x)
                c, o = c.view(-1, 1).to(self.device), o.view(-1, 4).to(self.device)
                o, offset = o[confi.view(-1) != 0], offset[confi.view(-1) != 0]
                c, confi = c[confi.view(-1) != 2], confi[confi.view(-1) != 2]
                # evalution
                _c = c[confi.view(-1) == 1]
                out_correct += torch.sum(_c, dim=0)
                label_correct += torch.sum(confi, dim=0)
                # loss
                loss_c = self.loss_confi(c, confi)
                loss_o = self.loss_offset(o, offset)
                regular = net.Regular(self.net).regular_loss()  # L2正则
                loss = loss_c+loss_o+regular
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print("Proccessing: {}/{}".format(i, len(self.train_set)))
            iou = self.iou(o[-1], offset[-1])
            print("$ 测试集：[epoche] - {}  Loss:  {:.2f}  Recall:  {:.2f}%  Iou:  {:.2f}".format(epoche, loss.item(), (out_correct/label_correct*100).item(), iou.item()))
            # validation
            self.net.eval()
            _loss, _rec, _iou = self.validation()
            print("# 验证集：[epoche] - {}  Loss:  {:.2f}  Recall:  {:.2f}%  Iou:  {:.2f}".format(epoche, _loss.item(), _rec.item()*100, _iou.item()))
            epoche += 1
            torch.save(self.net, self.save_path)

    def validation(self):
        out_correct, label_correct = 0, 0
        for x, confi, offset in self.validation_set:
            x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
            c, o = self.net(x)
            c, o = c.view(-1, 1).to(self.device), o.view(-1, 4).to(self.device)
            loss_c = self.loss_confi(c, confi)
            loss_o = self.loss_offset(o, offset)
            # evalution
            _c = c[confi.view(-1) == 1]
            out_correct += torch.sum(_c, dim=0)
            label_correct += torch.sum(confi, dim=0)
            rec = out_correct/label_correct
            iou = self.iou(o[-1], offset[-1])
            # loss
            loss = loss_c+loss_o
            return loss, rec, iou
            
    def iou(self, o, offset):
        x1, y1, x2, y2 = o.data*self.img_size
        x1, y1, x2, y2 = -x1, -y1, self.img_size-x2, self.img_size-y2
        _x1, _y1, _x2, _y2 = offset.data*self.img_size
        _x1, _y1, _x2, _y2 = 0-_x1, 0-_y1, self.img_size-_x2, self.img_size-_y2
        inter_area = (min(x2, _x2)-max(x1, _x1))*(min(y2, _y2)-max(y1, _y1))
        union_area = (x2-x1)*(y2-y1)+(_x2-_x1)*(_y2-_y1)-inter_area
        return inter_area/union_area






            
                



            



        


