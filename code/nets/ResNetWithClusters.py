# -*- coding: utf-8 -*-
# @Time : 2022/8/22 15:27
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
import torch
from torch import nn
import torch.nn.functional as F
from .PreResNet import PreActBlock, PreActBottleneck, BasicBlock, Bottleneck, conv3x3


class MultipleTaskNetwork(nn.Module):
    def __init__(self, backbone_name, num_classes=10, z_dim=128, n_clusters=10, norm_p=2, normalize=True):
        super(MultipleTaskNetwork, self).__init__()

        # Parameters
        self.norm_p = norm_p
        self.normalize = normalize

        # Backbone
        backbone, feature_dim = globals()[backbone_name]()
        self.backbone = backbone

        # Supervised prediction
        self.linear = nn.Linear(feature_dim, num_classes)

        # Pre-feature
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim, 4096), nn.BatchNorm1d(4096), nn.ReLU())

        # Cluster.
        self.cluster = nn.Linear(4096, n_clusters)
        # Subspace.
        self.subspace = nn.Linear(4096, z_dim)

    def forward(self, x, return_info='return_supervised_only'):
        feature = self.backbone(x)

        if return_info == 'return_supervised_only':
            predict_logits = self.linear(feature)
            return predict_logits

        elif return_info == 'return_supervised_and_clustering':
            predict_logits = self.linear(feature)

            pre_feature = self.pre_feature(feature)
            cluster_logits = self.cluster(pre_feature)

            return predict_logits, cluster_logits

        elif return_info == 'return_supervised_and_subspace':
            predict_logits = self.linear(feature)

            pre_feature = self.pre_feature(feature)
            z = self.subspace(pre_feature)
            if self.normalize:
                z = F.normalize(z, p=self.norm_p)

            return predict_logits, z

        elif return_info == 'return_supervised_and_clustering_and_subspace':
            predict_logits = self.linear(feature)

            pre_feature = self.pre_feature(feature)
            cluster_logits = self.cluster(pre_feature)
            z = self.subspace(pre_feature)
            if self.normalize:
                z = F.normalize(z, p=self.norm_p)

            return predict_logits, cluster_logits, z

        elif return_info == 'return_feature_only':
            return feature

        elif return_info == 'return_pre_feature_only':
            pre_feature = self.pre_feature(feature)
            return pre_feature

        elif return_info == 'return_clustering_only':
            pre_feature = self.pre_feature(feature)
            cluster_logits = self.cluster(pre_feature)
            return cluster_logits

        elif return_info == 'return_subspace_only':
            pre_feature = self.pre_feature(feature)
            z = self.subspace(pre_feature)
            if self.normalize:
                z = F.normalize(z, p=self.norm_p)
            return z

        elif return_info == 'return_clustering_and_subspace':
            pre_feature = self.pre_feature(feature)
            cluster_logits = self.cluster(pre_feature)
            z = self.subspace(pre_feature)
            if self.normalize:
                z = F.normalize(z, p=self.norm_p)
            return cluster_logits, z

        else:
            raise ValueError('Unknown return_info: {}'.format(return_info))

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_linear_supervised(self):
        for param in self.linear.parameters():
            param.requires_grad = False

    def unfreeze_linear_supervised(self):
        for param in self.linear.parameters():
            param.requires_grad = True

    def freeze_supervised(self):
        self.freeze_linear_supervised()
        self.freeze_backbone()

    def unfreeze_supervised(self):
        self.unfreeze_linear_supervised()
        self.unfreeze_backbone()


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, block, num_blocks, conv1_default=3):
        super(ResNetFeatureExtractor, self).__init__()
        self.in_planes = 64

        if conv1_default == 3:
            self.conv1 = conv3x3(3, 64)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out


def pre_resnet18():
    return ResNetFeatureExtractor(PreActBlock, [2, 2, 2, 2]), 512


def resnet18():
    return ResNetFeatureExtractor(BasicBlock, [2, 2, 2, 2]), 512


def pre_resnet34():
    return ResNetFeatureExtractor(PreActBlock, [3, 4, 6, 3]), 512


def resnet34():
    return ResNetFeatureExtractor(BasicBlock, [3, 4, 6, 3]), 512


class ResNetFeatureExtractorForClothes1M(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetFeatureExtractorForClothes1M, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out


def resnet50():
    return ResNetFeatureExtractorForClothes1M(Bottleneck, [3, 4, 6, 3]), 2048


def resnet101():
    return ResNetFeatureExtractorForClothes1M(Bottleneck, [3, 4, 23, 3]), 2048
