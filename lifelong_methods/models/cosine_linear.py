'''
Adapted from
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py

Reference:
[1] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, Dahua Lin
    Learning a Unified Classifier Incrementally via Rebalancing. CVPR 2019
'''

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from typing import Union


class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma: Union[bool, float, int] = True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features), requires_grad=True)
        if isinstance(sigma, bool):
            if sigma:
                self.sigma = Parameter(torch.empty(1), requires_grad=True)
                self.sigma.data.fill_(1)
            else:
                self.register_parameter('sigma', None)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.register_buffer('sigma', torch.tensor(float(sigma)))
        else:
            raise ValueError("sigma should be a boolean or a float")
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input_: torch.Tensor):
        out = F.linear(F.normalize(input_, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, sigma: Union[bool, float, int] = True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if isinstance(sigma, bool):
            if sigma:
                self.sigma = Parameter(torch.empty(1), requires_grad=True)
                self.sigma.data.fill_(1)
            else:
                self.register_parameter('sigma', None)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.register_buffer('sigma', torch.tensor(float(sigma)))
        else:
            raise ValueError("sigma should be a boolean or a float")

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = torch.cat((out1, out2), dim=1) # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out
        return out
