# -*- coding:utf-8 -*-
import os
import net
import torch
import dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Train:
    def __init__(self, parameter:str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameter = parameter
        self.net = net.LstmNet().to(self.device)
        if os.path.exists(parameter):
            print("Loading ... ...")
            self.net.load_state_dict(torch.load(parameter))
        self.train = dataset.train
        self.test = dataset.test
        self.opt = optim.Adam(self.net.parameters())
        self.loss_fun = nn.CrossEntropyLoss()

    def main(self):
        # _loss, _accuracy = [], []
        for epoche in range(1000):
            for i, (x, t) in enumerate(self.train):
                self.net.train()
                x, t = x.to(self.device), t.to(self.device)
                output = self.net(x).to(self.device)
                loss = self.loss_fun(output, t)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % 10 == 0:
                    print("[epoche] - {}: MODE: Train - Loss: {}".format(epoche, loss.item()))
                    # _loss.append(loss.item())
                    self.net.eval()
                    with torch.no_grad():
                        for xs, ys in self.test:
                            xs, ys = xs.to(self.device), ys.to(self.device)
                            y = self.net(xs)
                            acc = torch.mean(torch.argmax(y, dim=1) == ys, dtype=torch.float)
                            print("[epoche] - {}: MODE: Test - Loss: {}%".format(epoche, acc.item()*100))
                            # _accuracy.append(acc.item())
                            # plt.clf()
                            # plt.plot(_loss, "red")
                            # plt.plot(_accuracy, "green")
                            # plt.pause(0.001)
                            break
            torch.save(self.net.state_dict(), self.parameter)


if __name__ == '__main__':
    para = "G:/Project/Code/RNN/MNIST/lstm.pkl"
    trainer = Train(para)
    trainer.main()


