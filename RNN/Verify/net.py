# -*- coding:utf-8 -*-
import os
import torch
import dataset
import torch.nn as nn


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.linear = nn.Sequential(
                                    nn.Linear(in_features=dataset.SIZE[1]*dataset.SIZE[3], out_features=128),
                                    nn.BatchNorm1d(num_features=128),
                                    nn.ReLU()
                                    )
        self.gru = nn.GRU(128, 128, 4, batch_first=True)

    def forward(self, x):
        input_linear = x.permute(0, 3, 1, 2).reshape(shape=(-1, x.size(1)*x.size(2)))
        output_linear = self.linear(input_linear)
        input_gru = output_linear.reshape(shape=(-1, dataset.SIZE[2], 128))
        output_gru, _ = self.gru(input_gru)
        output = output_gru[:, -1:, :]
        return output


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.gru = nn.GRU(128, 128, 4, batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        input_gru = x.expand(x.size(0), dataset.LEVEL, x.size(-1))
        output_gru, _ = self.gru(input_gru)
        input_linear = output_gru.reshape(shape=(-1, output_gru.size(-1)))
        output_linear = self.linear(input_linear)
        output = output_linear.reshape(shape=(-1, dataset.LEVEL, output_linear.size(-1)))
        return output


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def forward(self, x):
        output_encoder = self.encoder(x)
        output_decoder = self.decoder(output_encoder)
        return output_decoder


class UnionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
                                    nn.Linear(in_features=180, out_features=128),
                                    nn.BatchNorm1d(num_features=128),
                                    nn.ReLU()
                                    )
        self.lstm = nn.GRU(input_size=128, hidden_size=128, num_layers=5, batch_first=True)
        self.classifier = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        input_linear = x.permute(0, 3, 1, 2).reshape(shape=(-1, x.size(1)*x.size(2)))
        output_linear = self.linear(input_linear)
        input_lstm = output_linear.view(x.size(0), -1, 128)
        output_lstm, _ = self.lstm(input_lstm)  # 省略h0, c0

        input = output_lstm[:, -1:, :]
        input_lstm2 = input.expand(x.size(0), 5, 128)
        output_lstm2, _ = self.lstm(input_lstm2)  # 省略h0, c0
        input_cls = output_lstm2.reshape(shape=(-1, 128))
        output_cls = self.classifier(input_cls)
        output = output_cls.view(x.size(0), -1, 10)
        return output


# if __name__ == '__main__':
#     x = torch.Tensor(10, 3, 60, 120)
#     net = MainNet()
#     net = UnionNet()
#     print(net(x).size())