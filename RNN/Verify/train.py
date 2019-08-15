# -*- coding:utf-8 -*-
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class Train:
    def __init__(self, save_path: str, img_path: str):
        self.save_path = save_path
        self.device = torch.device("cuda")
        self.net = net.MainNet(save_path).to(self.device)
        self.test = data.DataLoader(dataset.MyData(img_path + "/test"), batch_size=256, shuffle=True)
        self.train = data.DataLoader(dataset.MyData(img_path+"/train"), batch_size=256, shuffle=True, num_workers=4)
        self.optimize = optim.Adam([{"params": self.net.encoder.parameters()},
                                    {"params": self.net.decoder.parameters()}])
        self.loss = nn.CrossEntropyLoss()

    def main(self):
        for epoche in range(120):
            self.net.train()
            precision = []
            for i, (x, t) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                output = self.net(x)
                loss = self.loss(output.view(-1, 10), t.view(-1).long())
                self.optimize.zero_grad()
                loss.backward()
                self.optimize.step()
                predict = torch.argmax(output, dim=-1)
                acc = torch.mean((predict == t.long()).all(dim=-1).float())
                precision.append(acc.item())
            print("[epoche] - {} Loss:{} AP: {:.2f}%".format(epoche, loss.item(), sum(precision)/len(precision)*100))
            with torch.no_grad():
                self.net.eval()
                for _x, _t in self.test:
                    _x, _t = _x.to(self.device), _t.to(self.device)
                    _output = torch.softmax(self.net(_x), dim=-1)
                    _predict = torch.argmax(_output, dim=-1)
                    _acc = torch.mean((_predict == _t.long()).all(dim=-1).float())
                    print("Predict: {}\nTruth: {}\nAccuracy: {:.2f}%".format([k.item() for k in _predict[-1]],
                                                                         [z.item() for z in _t[-1]],
                                                                         _acc.item()*100))
                    break
            torch.save(self.net.encoder.state_dict(), self.save_path + "/encoder.pkl")
            torch.save(self.net.decoder.state_dict(), self.save_path+"/decoder.pkl")


if __name__ == '__main__':
    save_path = "G:/Project/Code/RNN/Verify/model"
    img_path = "F:/RNN/VERIFY"
    mytrain = Train(save_path, img_path)
    mytrain.main()


