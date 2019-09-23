# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, save_path: str, img_path: str):
        self.save_path = save_path
        self.device = torch.device("cuda")
        self.net = net.MainNet().to(self.device)
        if os.path.exists(save_path):
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(save_path))
        self.summarywriter = SummaryWriter(log_dir="G:/Project/Code/RNN/Verify/runs")
        self.test = data.DataLoader(dataset.MyData(img_path + "/test"), batch_size=256, shuffle=True)
        self.train = data.DataLoader(dataset.MyData(img_path+"/train"), batch_size=256, shuffle=True, num_workers=4)
        self.optimize = optim.Adam(self.net.parameters(), weight_decay=1e-4)
        self.loss = nn.CrossEntropyLoss()
    def main(self):
        for epoche in range(1000):
            self.net.train()
            for i, (x, t) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                output = self.net(x)
                loss = self.loss(output.view(-1, 10), t.view(-1).long())
                self.optimize.zero_grad()
                loss.backward()
                self.optimize.step()
                self.summarywriter.add_scalar("Train Loss", loss, global_step=i)

            with torch.no_grad():
                self.net.eval()
                for _x, _t in self.test:
                    _x, _t = _x.to(self.device), _t.to(self.device)
                    _output = torch.softmax(self.net(_x), dim=-1)
                    _loss = self.loss(_output.view(-1, 10), _t.view(-1).long())
                    _predict = torch.argmax(_output, dim=-1)
                    _acc = torch.mean((_predict == _t.cuda().long()).all(dim=1).float())
                    print("Predict: {}\nTruth: {}\nAccuracy: {:.2f}%".format([k.item() for k in _predict[-1]],
                                                                             [z.item() for z in _t[-1]],
                                                                             _acc.item()*100))
                    break
            self.summarywriter.add_scalar("Test Loss", _loss, global_step=epoche)
            torch.save(self.net.state_dict(), self.save_path)


if __name__ == '__main__':
    save_path = "G:/Project/Code/RNN/Verify/main4.pkl"
    img_path = "F:/RNN/VERIFY"
    mytrain = Train(save_path, img_path)
    mytrain.main()


