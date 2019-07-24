# -*-coding:utf-8-*-
import torch
import torch.nn as nn


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.LeakyReLU(), 
                                            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
                                            nn.LeakyReLU())

        self.out_1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1),
                                            nn.Sigmoid())
        self.out_2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1)

    def forward(self, x):
        h = self.layer(x)

        y_1 = self.out_1(h)
        y_2 = self.out_2(h)

        return y_1, y_2

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.LeakyReLU(), 
                                            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.MaxPool2d(kernel_size=3, stride=2),
                                            nn.LeakyReLU(), 
                                            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1),
                                            nn.LeakyReLU())
        self.layer_2 = nn.Sequential(nn.Linear(in_features=64*3*3, out_features=128, bias=True),
                                            nn.LeakyReLU())

        self.out_1 = nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True),
                                            nn.Sigmoid())
        self.out_2 = nn.Linear(in_features=128, out_features=4, bias=True)

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
        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                            nn.LeakyReLU(), 
                                            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.MaxPool2d(kernel_size=3, stride=2),
                                            nn.LeakyReLU(), 
                                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                            nn.LeakyReLU(),
                                            nn.MaxPool2d(kernel_size=2, stride=2),
                                            nn.LeakyReLU(), 
                                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
                                            nn.LeakyReLU())
        self.layer_2 = nn.Sequential(nn.Linear(in_features=128*3*3, out_features=256, bias=True),
                                            nn.LeakyReLU())

        self.out_1 = nn.Sequential(nn.Linear(in_features=256, out_features=1, bias=True),
                                            nn.Sigmoid())
        self.out_2 = nn.Linear(in_features=256, out_features=4, bias=True)

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
    def __init__(self, model, weight_decay=0.01, p=2):
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

# if __name__ == "__main__":
#     pnet = PNet()
#     rnet = RNet()
    # onet = ONet()
    # net = torch.load("F:/Project/Code/MTCNN/pnet.pth")
    # import PIL.Image as Image
    # import torchvision.transforms as tf 
    # a = 0
    # b = 1
    # for i in range(10000):
    #     try:
    #         img = Image.open("F:/Project/DataSet/celebre/24/positive/{}.jpg".format(i))
    #         data = tf.ToTensor()(img).unsqueeze(dim=0)
    #         c, o = net(data.cuda())
    #         if c.item() < 0.1:
    #             a += 1
    #         b += 1
    #     except:
    #         continue
    # print("Accuracy rate: {} / {} = {}".format(a, b, 1-a/b))
    # loss = Regular(onet)
    # print(loss.regular_loss())
   

    


