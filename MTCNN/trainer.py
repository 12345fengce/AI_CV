# -*-coding:utf-8-*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class Trainer:
    def __init__(self, model, train, validation, save, size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save = save
        self.size = size
        if os.path.exists(save):
            print("loading ... ...")
            self.net = self.model.load_state_dict(torch.load(save))
        else:
            print("creating ... ...")
            self.net = model.to(self.device)
        self.train_set = data.DataLoader(dataset.MyData(train, size), batch_size=512, shuffle=True, num_workers=3)
        self.validation_set = data.DataLoader(dataset.MyData(validation, size), batch_size=128, shuffle=True)
        self.optimize = optim.Adam(self.net.parameters())
        self.loss_confi = nn.BCELoss()
        self.loss_offset = nn.MSELoss()

    def main(self):
        epoche = 1
        while True:
            # mode:train
            self.net.train()
            print("[epoche] - {}:".format(epoche))
            out_positive, label_positive = 0, 0
            for i, (x, confi, offset) in enumerate(self.train_set):
                x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
                _confi, _offset = self.net(x)
                _confi, _offset = _confi.view(-1, 1), _offset.view(-1, 14)
                _offset, offset = _offset[confi.view(-1) != 0], offset[confi.view(-1) != 0]
                _confi, confi = _confi[confi.view(-1) != 2], confi[confi.view(-1) != 2]
                # recall rate
                c = _confi[confi.view(-1) == 1]
                out_positive += torch.sum(c, dim=0)
                label_positive += torch.sum(confi, dim=0)
                # loss
                loss_c = self.loss_confi(_confi, confi)
                loss_o = self.loss_offset(_offset, offset)
                # loss_regular = net.Regular(self.net).regular_loss().to(self.device)  # L2正则
                loss = loss_c+loss_o
                self.optimize.zero_grad()
                loss.backward()
                self.optimize.step()
                print("Proccessing: {}/{}".format(i, len(self.train_set)))
                iou = self.iou(_offset[-1], offset[-1])
                print("$ 测试集：[epoche] - {}  Loss:  {:.2f}  Recall:  {:.2f}%  Iou:  {:.2f}"
                      .format(epoche, loss.item(), (out_positive/label_positive*100).item(), iou.item()))
            epoche += 1
            torch.save(self.net.state_dict, self.save)
            if epoche == 500:
                # mode:verify
                self.net.eval()
                _loss, _rec, _iou = self.verify()
                print("# 验证集：[epoche] - {}  Loss:  {:.2f}  Recall:  {:.2f}%  Iou:  {:.2f}"
                      .format(epoche, _loss.item(), _rec.item()*100, _iou.item()))

    def verify(self):
        out_positive, label_positive = 0, 0
        for x, confi, offset in self.validation_set:
            x, confi, offset = x.to(self.device), confi.to(self.device), offset.to(self.device)
            _confi, _offset = self.net(x)
            _confi, _offset = _confi.view(-1, 1), _offset.view(-1, 14)
            _offset, offset = _offset[confi.view(-1) != 0], offset[confi.view(-1) != 0]
            _confi, confi = _confi[confi.view(-1) != 2], confi[confi.view(-1) != 2]
            # recall rate
            c = _confi[confi.view(-1) == 1]
            out_positive += torch.sum(c, dim=0)
            label_positive += torch.sum(confi, dim=0)
            rec = out_correct / label_correct
            # loss
            loss_c = self.loss_confi(_confi, confi)
            loss_o = self.loss_offset(_offset, offset)
            loss = loss_c + loss_o
            # iou
            iou = self.iou(_offset[-1], offset[-1])
            return loss, rec, iou
            
    def iou(self, _offset, offset):
        _x1, _y1, _x2, _y2, *_landmark = _offset.data * self.size
        _x1, _y1, _x2, _y2 = 0 - _x1, 0 - _y1, self.size - _x2, self.size - _y2
        x1, y1, x2, y2, *landmark = offset*self.size
        x1, y1, x2, y2 = 0-x1, 0-y1, self.size-x2, self.size-y2  # label
        inter_area = (min(x2, _x2)-max(x1, _x1))*(min(y2, _y2)-max(y1, _y1))
        union_area = (x2-x1)*(y2-y1)+(_x2-_x1)*(_y2-_y1)-inter_area
        return inter_area/union_area






            
                



            



        


