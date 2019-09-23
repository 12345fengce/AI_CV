# -*-coding:utf-8-*-
import torch
import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.layer = nn.Sequential(
                                    nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),
                                    nn.LeakyReLU(0.1),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
                                    nn.LeakyReLU(0.1),
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                                    nn.LeakyReLU(0.1)
                                    )

        self.out_1 = nn.Sequential(
                                    nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1),
                                    nn.Sigmoid()
                                    )
        self.out_2 = nn.Conv2d(in_channels=32, out_channels=14, kernel_size=1, stride=1)

    def forward(self, x):
        h = self.layer(x)
        y_1 = self.out_1(h)
        y_2 = self.out_2(h)
        return y_1, y_2


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.layer_1 = nn.Sequential(
                                        nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1),
                                        nn.LeakyReLU(0.1),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),
                                        nn.LeakyReLU(0.1),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),
                                        nn.LeakyReLU(0.1)
                                        )
        self.layer_2 = nn.Sequential(
                                        nn.Linear(in_features=64*3*3, out_features=128, bias=True),
                                        nn.LeakyReLU(0.1)
                                        )

        self.out_1 = nn.Sequential(
                                    nn.Linear(in_features=128, out_features=1, bias=True),
                                    nn.Sigmoid()
                                    )
        self.out_2 = nn.Linear(in_features=128, out_features=14, bias=True)

    def forward(self, x):
        h = self.layer_1(x)
        h = h.reshape(shape=(h.size(0), -1))
        y = self.layer_2(h)
        y_1 = self.out_1(y)
        y_2 = self.out_2(y)
        return y_1, y_2


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.layer_1 = nn.Sequential(
                                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),  # 46
                                        nn.LeakyReLU(0.1),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 23
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),  # 21
                                        nn.LeakyReLU(0.1),
                                        nn.MaxPool2d(kernel_size=3, stride=2),  # 10
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 8
                                        nn.LeakyReLU(0.1),
                                        nn.MaxPool2d(kernel_size=2, stride=2),  # 4
                                        nn.LeakyReLU(0.1),
                                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),  # 3
                                        nn.LeakyReLU(0.1)
                                        )
        self.layer_2 = nn.Sequential(
                                        nn.Linear(in_features=128*3*3, out_features=256, bias=True),
                                        nn.LeakyReLU(0.1)
                                        )

        self.out_1 = nn.Sequential(
                                    nn.Linear(in_features=256, out_features=1, bias=True),
                                    nn.Sigmoid()
                                    )
        self.out_2 = nn.Linear(in_features=256, out_features=14, bias=True)

    def forward(self, x):
        h = self.layer_1(x)
        h = h.reshape(shape=(h.size(0), -1))
        y = self.layer_2(h)
        y_1 = self.out_1(y)
        y_2 = self.out_2(y)
        return y_1, y_2


class Regular(nn.Module):
    """只对W参数进行惩罚，bias不进行惩罚
        model: 网络
        weight_decay: 衰减系数（加权）
        p: 范数，默认L2"""
    def __init__(self, model, weight_decay=0, p=2):
        super(Regular, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p

    def regular_loss(self):
        regular_loss = 0
        for param in self.model.parameters():
            if len(param.size()) > 1:
                regular_loss += torch.norm(param, self.p)
        return regular_loss*self.weight_decay

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.Tensor(89, 3, 48, 48).cuda()
    onet = ONet().to(device)
    c, y = onet(x)
    print(c.shape, y.shape)
   

    


