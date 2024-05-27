# -*- coding: utf-8 -*-
# @Time : 2022/10/14 13:06
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

from torch import nn
from .losses.fmix import FMix
from .losses.loss_dividemix import NegEntropy, SemiLoss


class LossController:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.cross_entropy_loss = None
        self.fmix_module = None
        self.neg_entropy = None
        self.semi_loss = None

    def get_cross_entropy_loss(self):
        if self.cross_entropy_loss is None:
            self.cross_entropy_loss = nn.CrossEntropyLoss()

        return self.cross_entropy_loss

    @staticmethod
    def get_cross_entropy_loss_none_reduction():
        return nn.CrossEntropyLoss(reduction="none")

    def get_fmix_module(self):
        if self.fmix_module is None:
            self.fmix_module = FMix()

        return self.fmix_module

    def get_neg_entropy(self):
        if self.neg_entropy is None:
            self.neg_entropy = NegEntropy()

        return self.neg_entropy

    def get_semi_loss_for_cluster_mix(self, lambda_u=1):
        if self.semi_loss is None:
            self.semi_loss = SemiLoss(lambda_u=lambda_u)

        return self.semi_loss


