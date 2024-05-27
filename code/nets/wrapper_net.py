# -*- coding: utf-8 -*-
# @Time : 2022/10/14 11:10
# @Author : Mengtian Zhang
# @E-mail : zhangmengtian@sjtu.edu.cn
# @Version : v-dev-0.0
# @License : MIT License
# @Copyright : Copyright 2022, Mengtian Zhang
# @Function 

"""Summary.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

from .models.resnet import *


# ------------------------------------------------------------------------------
# get networks.

## get single network.
def get_single_network(net_name='pre_resnet34', num_classes=10):
    net = globals()[net_name](num_classes)

    return net


## get dual networks.
def get_dual_networks(net_name='pre_resnet34', num_classes=10):
    return DualNets(net_name, num_classes)


class DualNets(nn.Module):
    def __init__(self, net_name='pre_resnet34', num_classes=10):
        super(DualNets, self).__init__()
        self.net_name = net_name
        self.num_classes = num_classes

        self.net1 = get_single_network(net_name, num_classes)
        self.net2 = get_single_network(net_name, num_classes)

    def forward(self, x):
        output = (self.net1(x) + self.net2(x)) / 2
        return output

    def __iter__(self):
        return iter([self.net1, self.net2])
