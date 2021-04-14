import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

#from models.layers import CBR
#from models.models_utils import weights_init, print_network

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
class CBR(nn.Module):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=nn.ReLU(True), dropout=False):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        if sample=='down':
            self.c = nn.Conv2d(ch0, ch1, 4, 2, 1)
        else:
            self.c = nn.ConvTranspose2d(ch0, ch1, 4, 2, 1)
        if bn:
            self.batchnorm = nn.BatchNorm2d(ch1, affine=True)
        if dropout:
            self.Dropout = nn.Dropout()

    def forward(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = self.Dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class _Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = in_ch

        self.c0_0 = CBR(in_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c0_1 = CBR(out_ch, 32, bn=False, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c1 = CBR(64, 128, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c2 = CBR(128, 256, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c3 = CBR(256, 512, bn=True, sample='down', activation=nn.LeakyReLU(0.2, True), dropout=False)
        self.c4 = nn.Conv2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x_0 = x[:, :self.in_ch]
        x_1 = x[:, self.in_ch:]
        h = torch.cat((self.c0_0(x_0), self.c0_1(x_1)), 1)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids

        self.dis = nn.Sequential(OrderedDict([('dis', _Discriminator(in_ch, out_ch))]))

        self.dis.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.dis, x, self.gpu_ids)
        else:
            return self.dis(x)