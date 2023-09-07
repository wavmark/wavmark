import torch
import torch.nn as nn
from ..models.rrdb_denselayer import ResidualDenseBlock_out


class INV_block(nn.Module):
    def __init__(self, channel=2, subnet_constructor=ResidualDenseBlock_out, clamp=2.0):
        super().__init__()
        self.clamp = clamp

        # ρ
        self.r = subnet_constructor(channel, channel)
        # η
        self.y = subnet_constructor(channel, channel)
        # φ
        self.f = subnet_constructor(channel, channel)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, rev=False):
        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return y1, y2
