import torch
from ..models.invblock import INV_block


class Hinet(torch.nn.Module):

    def __init__(self, in_channel=2, num_layers=16):
        super(Hinet, self).__init__()
        self.inv_blocks = torch.nn.ModuleList([INV_block(in_channel) for _ in range(num_layers)])

    def forward(self, x1, x2, rev=False):
        # x1:cover
        # x2:secret
        if not rev:
            for inv_block in self.inv_blocks:
                x1, x2 = inv_block(x1, x2)
        else:
            for inv_block in reversed(self.inv_blocks):
                x1, x2 = inv_block(x1, x2, rev=True)
        return x1, x2
