import torch
import torch.nn as nn
from ..models import module_util as mutil


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(in_channel + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_channel + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_channel + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(in_channel + 4 * 32, out_channel, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
