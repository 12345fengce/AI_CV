# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt


class Train:
    def __init__(self, params: str):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = net.SignNet().to(self.device)
        if os.path.exists(params):
            self.model.load_state_dict(torch.load(params))
        self.train = data.DataLoader(dataset.MyData(dataset.PATH), batch_size=1, shuffle=False)
        self.optim = optim.Adam(self.model.parameters())
        self.loss = nn.BCELoss()

    def run(self, epoche):
        for i in range(epoche):
            seq, predict, truth = [], [], []
            for j, (x, t, m) in enumerate(self.train):
                x, t = x.to(self.device), t.to(self.device)
                y = self.model(x)
                loss = self.loss(y, t)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                seq.append(j)
                predict.append(y.item()*int(m))
                truth.append(t.item()*int(m))
                # # plt.clf()
                # plt.title("[eopche] - {} - Signal Loss:{}".format(i, loss.item()))
                # plt.ylim(0, 1e+10)
                # plt.plot(seq, predict, ls="-", color="g", label="Pre")
                # plt.plot(seq, truth, ls="--", color="r", label="True")
                # plt.pause(0.1)
                print("[epoche] - {} - Loss: {}".format(i, loss.item()))
            print(len(predict))
            torch.save(self.model.state_dict(), self.params)


if __name__ == '__main__':
    params = "./signnet.pkl"
    mytrain = Train(params)
    mytrain.run(100)



