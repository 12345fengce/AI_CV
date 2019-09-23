# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class LstmNet(nn.Module):
    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=64, out_features=10)

    def forward(self, input):
        """input shape of (N, C, H, W)
            expection rnn shape of (N, S, V)"""
        input = input.reshape(shape=(input.size(0), -1, input.size(-1)))  # 将图片每行像素输入到RNN中
        # c_0 = h_0 = torch.zeros(2*1, input.size(0), 64)
        output_lstm, _ = self.lstm(input)
        print(output_lstm.size())
        input_linear = output_lstm[:, -1, :]  # (N, S, V) get output of the last S
        output_linear = self.linear(input_linear)
        return output_linear


# if __name__ == '__main__':
#     lstm = LstmNet()
#     input = torch.rand((1, 3, 28, 28))
#     output = lstm(input)
#     print(output.size())
