# -*- coding: utf-8 -*-
# @Time : 2022/8/21 16:31
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
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LossControl:
    def __init__(self, lambda_u=0):
        super(LossControl, self).__init__()

        self.semi_loss = SemiLoss(lambda_u=lambda_u)
        self.cross_entropy_none = nn.CrossEntropyLoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.conf_penalty = NegEntropy()

        self.gumble_softmax = None
        self.mcrr_loss = None

    def register_gumble_softmax(self, tau):
        self.gumble_softmax = GumbleSoftmax(tau)

    def register_mcrr_loss(self, eps=0.5, gamma=1.0):
        self.mcrr_loss = MaximalCodingRateReduction(eps, gamma)


def linear_ramp_up(current, start, ramp_up_length=16):
    current = np.clip((current - start) / ramp_up_length, 0.0, 1.0)
    return float(current)


class SemiLoss:
    def __init__(self, lambda_u=0):
        super(SemiLoss, self).__init__()

        self.lambda_u = lambda_u

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.lambda_u * linear_ramp_up(epoch, warm_up)


class NegEntropy:
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


class GumbleSoftmax(nn.Module):
    def __init__(self, tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through

    def forward(self, logits):
        log_ps = torch.log_softmax(logits, dim=1)
        gumble = torch.rand_like(log_ps).log().mul(-1).log().mul(-1)
        logits = log_ps + gumble
        out = (logits / self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits * 1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, eps=0.01, gamma=1.):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)

        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        # This function support Y as label integer or membership probability.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for idx, label in enumerate(Y):
                Pi[label, 0, idx] = 1
        else:
            # if Y is a probability matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        discrimn_loss = self.compute_discrimn_loss(W)
        compress_loss = self.compute_compress_loss(W, Pi)

        total_loss = - discrimn_loss + self.gamma * compress_loss
        return total_loss, [discrimn_loss.item(), compress_loss.item()]
