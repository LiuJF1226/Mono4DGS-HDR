import torch
import torch.nn as nn


class ToneMapper(nn.Module):
    def __init__(self, hidden, act):
        super().__init__()
        if act == "sp":
            self.activate = nn.Softplus()
        if act == "relu":
            self.activate = nn.ReLU()
        self.tm_r = nn.Sequential(nn.Linear(1, hidden), self.activate, nn.Linear(hidden, 1), nn.Sigmoid())
        self.tm_g = nn.Sequential(nn.Linear(1, hidden), self.activate, nn.Linear(hidden, 1), nn.Sigmoid())
        self.tm_b = nn.Sequential(nn.Linear(1, hidden), self.activate, nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        r = self.tm_r(x[:,0:1])
        g = self.tm_g(x[:,1:2])
        b = self.tm_b(x[:,2:3])
        y = torch.cat([r, g, b], dim=-1)
        return y   

