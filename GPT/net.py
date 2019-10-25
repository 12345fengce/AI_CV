# -*- coding:utf-8 -*-
import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """multi heads self attention"""

    def __init__(self, word_dim: int, head_num: int, isMask=False):
        super(Attention, self).__init__()
        self.word_dim = word_dim
        self.head_num = head_num
        self.isMask = isMask
        "dk"
        self.dk = word_dim
        "Q、K、V"
        self.separate = nn.Linear(word_dim, word_dim * 3)
        "multi heads"
        self.multi = nn.Linear(word_dim * 3, word_dim * 3 * head_num)
        "word restore"
        self.restore = nn.Linear(word_dim * head_num, word_dim)
        "mask"
        if isMask:
            self.register_buffer("mask", torch.tril(torch.ones(word_dim, word_dim)))

    def forward(self, x):
        x = self.separate(x)  # to (N, S, V)
        x = self.multi(x)

        x = x.reshape(*x.shape[:-1], self.head_num, -1)  # to (N, head_num, S, V)
        x = x.transpose(-2, -3)

        q, k, v = torch.chunk(x, 3, dim=-1)  # to (N, head_num, S, S)
        w = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)

        if self.isMask:
            mask = mask[:w.size(-2)][:w.size(-1)]
            w = w * mask - (1 - mask) * 1e5
        w = F.softmax(w, dim=-1)

        a = w @ v  # to (N, head_num, S, V)

        a = a.transpose(-2, -3)  # to (N, S, V)
        a = a.reshpe(*a.shape[:-2], -1)

        return self.restore(a)





