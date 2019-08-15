# -*- coding:utf-8 -*-
import os
import torch
import dataset
import torch.nn as nn


class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.linear = nn.Linear(in_features=dataset.SIZE[1], out_features=128)
        self.gru = nn.GRU(128, dataset.SIZE[1], 1, batch_first=True)

    def forward(self, x):
        """Linear
            transform (N, v) to (N, S, V)
            GRU
            transfrom (N, S, V) to (N, 1, V)"""
        output_linear = self.linear(x)
        input_gru = output_linear.reshape(shape=(-1, dataset.SIZE[2]*dataset.SIZE[3], 128))
        output_gru, _ = self.gru(input_gru)
        output = output_gru[:, -1:, :]
        return output


class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.gru = nn.GRU(dataset.SIZE[1], 128, 1, batch_first=True)
        self.linear = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """GRU
            transform (N, S, V) to (N, V)
            Linear
            transform (N, V) to (N, S, V)"""
        output_gru, _ = self.gru(x)
        input_linear = output_gru.reshape(shape=(-1, output_gru.size(-1)))
        output_linear = self.linear(input_linear)
        output = output_linear.reshape(shape=(-1, dataset.LEVEL, output_linear.size(-1)))
        return output


class MainNet(nn.Module):
    def __init__(self, path):
        super(MainNet, self).__init__()
        self.encoder = EncoderNet()
        self.decoder = DecoderNet()
        if len(os.listdir(path)) > 2:
            self.encoder.load_state_dict(torch.load(path+"/encoder.pkl"))
            self.decoder.load_state_dict(torch.load(path+"/decoder.pkl"))

    def forward(self, x):
        """transform x to (N, V)
            EncoderNet
            transform (N, 1, V) to (N, LEVEL, V)
            DecoderNet
            transform (N, V) to (N, LEVEL, V)"""
        input_encoder = x.permute(0, 1, 3, 2).reshape(shape=(-1, x.size(-2)))
        output_encoder = self.encoder(input_encoder)
        input_decoder = output_encoder.expand(output_encoder.size(0), dataset.LEVEL, output_encoder.size(-1))
        output_decoder = self.decoder(input_decoder)
        return output_decoder


if __name__ == '__main__':
    x = torch.Tensor(1, 3, 60, 120)
    PATH = "G:/Project/Code/RNN/Verify/model"
    net = MainNet(PATH)
    print(net(x).size())