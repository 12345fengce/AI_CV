# -*- coding:utf-8 -*-
import torch
import random


R = torch.tensor([[-1, -1, -1, -1, 0, -1],
                   [-1, -1, -1, 0, -1, 1],
                   [-1, -1, -1, 0, -1, -1],
                   [-1, 0, 0, -1, 0, -1],
                   [0, -1, -1, 0, -1, 1],
                   [-1, 0, -1, -1, 0, 1],])

V = torch.zeros(R.size())
γ = 0.9


# Bellman: Qt = rt + γ* max(Qt+1)
for step in range(10000):
    i, j = random.randint(0, R.size(0)-1), random.randint(0, R.size(0)-1)
    V[i, j] = R[i, j]+γ*max(V[j, :])  # γ系数0-1之间
    if step % 10 == 0:
        print("V", V)
