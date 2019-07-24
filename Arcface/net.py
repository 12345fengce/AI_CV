# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# Net for MNIST
class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(Convolution, self).__init__()
        self.Convolution = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                                                nn.BatchNorm2d(out_channels),
                                                                nn.PReLU())
    def forward(self, x):
        return self.Convolution(x)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Sequential(Convolution(1, 32, 3, 1, 0),
                                                    Convolution(32, 64, 3, 2, 1, 0),
                                                    Convolution(64, 32, 1, 1, 0),
                                                    Convolution(32, 32, 3, 1, 0),
                                                    Convolution(32, 64, 1, 1, 0),
                                                    Convolution(64, 128, 3, 2, 0),
                                                    Convolution(128, 64, 1, 1, 0),
                                                    Convolution(64, 64, 3, 1, 0),
                                                    Convolution(64, 64, 3, 1, 0),
                                                    Convolution(64, 128, 1, 1, 0))
        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=64, bias=False),
                                                    nn.BatchNorm1d(64),
                                                    nn.PReLU())
        self.features = nn.Linear(in_features=64, out_features=2, bias=False)  # margin
        self.outputs = nn.Linear(in_features=2, out_features=10, bias=False)  # softmax
    def forward(self, x):
        c = self.conv(x)
        c = torch.reshape(c, shape=(c.size(0), -1))
        l = self.linear(c)
        features = self.features(l)
        outputs = self.outputs(features)
        return features, outputs

# if __name__ == "__main__":
#     x = torch.Tensor(2, 1, 28, 28)
#     net = MyNet()
#     f, o = net(x)
#     print(f.size(), o.size())


# CenterLoss
class CenterLoss(nn.Module):
    """CentreLoss convert data and labels transforms to loss
        cls_num, feature_num: int
        x: torch.Tensor  labels: torch.tensor ndim=1"""
    def __init__(self, cls, features):
        super(CenterLoss, self).__init__()
        self.cls = cls
        self.centre = nn.Parameter(torch.randn(cls, features))
    def forward(self, x, labels):
        centre = self.centre[labels]
        count = torch.histc(labels.float(), bins=self.cls, min=min(labels), max=max(labels))[labels]
        distance = torch.sum(torch.sum((x.float()-centre.float())**2, dim=1)/count.float())
        return torch.sqrt(distance)

# if __name__ == "__main__":
#     x = torch.Tensor([[1, 3], [2, 4], [5, 7]])
#     label = torch.tensor([0, 1, 0])
#     feature_num, cls_num = 2, 2
#     loss = CenterLoss(cls_num, feature_num)
#     print(loss(x, label))


# ArcSoftmaxLoss
class ArcLoss(nn.Module):
    """ArcSoftmaxLoss to instead Softmax
        feature_num, cls_num: int
        X: shape=(n, v)  W: shape=(v, o)"""
    def __init__(self, features):
        super(ArcLoss, self).__init__()
        self.W = nn.Parameter(torch.randn(features))
    def forward(self, X, alpha=0.1):
        x, w = torch.norm(X, dim=1), torch.norm(self.W, dim=0)
        cosa = torch.matmul(X, self.W)/(x*w)
        a = torch.acos(cosa)
        molecule = torch.exp(x*w*torch.cos(a+alpha))
        denominator = torch.sum(torch.exp(x*w*cosa))-torch.exp(x*w*cosa)+molecule
        return molecule/denominator

# if __name__ == "__main__":
#     loss = ArcLoss(784)
#     x = torch.randn(10, 784)
#     alpha = 0.1
#     print(torch.sum(loss(x, alpha)))





