# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# CentreLoss
# create data,labels,centre
data = torch.tensor([[1, 2], [3, 4], [5, 6]])
labels = torch.tensor([0, 0, 1])
centre = torch.tensor([[1, 1], [3, 3]])
# got the same shape
centre = centre[labels]
# calculate distance between data and centre
distance = torch.sqrt(torch.sum((data.float()-centre.float())**2, dim=1))
loss = torch.sum(distance)
# print(loss)

class CentreLoss(nn.Module):
    """CentreLoss convert data and labels transforms to loss
        cls_num, feature_num: int
        x: torch.Tensor  labels: torch.tensor ndim=1"""
    def __init__(self, feature_num, cls_num):
        super(CentreLoss, self).__init__()
        self.centre = nn.Parameter(torch.randn((cls_num, feature_num)))

    def forward(self, x, labels):
        print(self.centre)
        centre = self.centre[labels]
        distance = torch.sqrt(torch.sum((x.float()-centre.float())**2, dim=1))
        return torch.sum(distance)


# ArcSoftmaxLoss
class ArcLoss(nn.Module):
    """ArcSoftmaxLoss to instead Softmax
        feature_num, cls_num: int
        X: shape=(n, v)  W: shape=(v, o)"""
    def __init__(self, feature_num):
        super(ArcLoss, self).__init__()
        self.W = nn.Parameter(torch.randn(feature_num))

    def forward(self, X, α):
        x, w = torch.norm(X, dim=1), torch.norm(self.W, dim=0)
        cosa = torch.matmul(X, self.W)/(x*w)
        a = torch.acos(cosa)
        molecule = torch.exp(x*w*torch.cos(a+α))
        denominator = torch.sum(torch.exp(x*w*cosa))-torch.exp(x*w*cosa)+molecule
        return molecule/denominator
loss = ArcLoss(784)
x = torch.randn(10, 784)
α = 0.1
print(torch.sum(loss.forward(x, α)))





