# -*- coding:utf-8 -*-
import torch
import dataset
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self):
        super(CBOW, self).__init__()
        self.lexicon = nn.Embedding(len(dataset.mydata.database), 16)
        self.linear_w1 = nn.Sequential(
                                        nn.Linear(16, 32),
                                        nn.ReLU(),
                                        )
        self.linear_w2 = nn.Sequential(
                                        nn.Linear(16, 32),
                                        nn.ReLU(),
                                        )
        self.linear_w3 = nn.Sequential(
                                        nn.Linear(16, 32),
                                        nn.ReLU(),
                                        )
        self.linear_w4 = nn.Sequential(
                                        nn.Linear(16, 32),
                                        nn.ReLU(),
                                        )
        self.linear_sum = nn.Linear(32, 16)

    def forward(self, x):
        w1 = self.linear_w1(self.lexicon(x[:, 0]))
        w2 = self.linear_w2(self.lexicon(x[:, 1]))
        w3 = self.linear_w3(self.lexicon(x[:, 3]))
        w4 = self.linear_w4(self.lexicon(x[:, 4]))
        w = w1+w2+w3+w4
        output = self.linear_sum(w)
        return output


# if __name__ == '__main__':
#     net = CBOW()
#     x = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 2, 1, 4]]).long()
#     print(net(x))