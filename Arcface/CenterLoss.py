# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# Net for MNIST
class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Convolution, self).__init__()
        self.Convolution = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                                        nn.BatchNorm2d(out_channels),
                                        nn.LeakyReLU(0.1),
                                        )

    def forward(self, data):
        return self.Convolution(data)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv = nn.Sequential(
                                    Convolution(1, 32, 1, 1),
                                    Convolution(32, 64, 3, 2),
                                    Convolution(64, 32, 1, 1),
                                    Convolution(32, 64, 3, 2),
                                    Convolution(64, 32, 1, 1),
                                    )
        self.linear = nn.Sequential(
                                    nn.Linear(32*6*6, 128),
                                    nn.BatchNorm1d(128),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(128, 2),
                                    )
        self.classify = nn.Linear(in_features=2, out_features=10, bias=False)

    def forward(self, x):
        output_conv = self.conv(x)
        input_linear = torch.reshape(output_conv, shape=(output_conv.size(0), -1))
        features = self.linear(input_linear)
        outputs = self.classify(features)
        return features, outputs

# if __name__ == "__main__":
#     x = torch.Tensor(2, 1, 28, 28)
#     net = MyNet()
#     f, o = net(x)
#     print(f.size(), o.size())


# CenterLoss


class CenterLoss(nn.Module):
    """CenterLoss convert data and labels transforms to loss
        cls_num, feature_num: int
        x: torch.Tensor  labels: torch.tensor ndim=1"""
    def __init__(self, cls, features):
        super(CenterLoss, self).__init__()
        self.cls = cls
        self.center = nn.Parameter(torch.randn(cls, features))

    def forward(self, x, labels):
        center = self.center[labels]  # center: ndim = 2  labels: ndim = 1  result: ndim = 2
        # bins + min = max
        count = torch.histc(labels.float(), bins=torch.max(labels)+1, min=0, max=torch.max(labels)+1)[labels]
        distance = torch.sum(torch.sqrt(torch.sum((x.float()-center.float())**2, dim=1))/count.float())
        return distance

# if __name__ == "__main__":
#     x = torch.Tensor([[1, 3], [2, 4], [5, 7]])
#     label = torch.tensor([0, 1, 0])
#     feature_num, cls_num = 2, 2
#     loss = CenterLoss(cls_num, feature_num)
#     print(list(loss.parameters()))


# ArcSoftmaxLoss










