# -*- coding: utf-8 -*-
# @Time : 2022/7/17 22:28
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

import sys
import os

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from tools import utils


def load_data_moon(path='./dataset_moon/sample-number-500/data_noise-rate-0.2900.pickle'):
    moon_data_dict = utils.load_pickle(path)

    data = moon_data_dict['data']
    label_true = moon_data_dict['label_true']
    label_noisy = moon_data_dict['label_noisy']

    return data, label_true, label_noisy


def get_dataset(path='./dataset_moon/sample-number-500/data_noise-rate-0.2900.pickle'):
    data, label_true, label_noisy = load_data_moon(path)
    dataset = MoonDataSet(data, label_true, label_noisy)

    return dataset


def get_dataset_division(path='./dataset_moon/sample-number-500/data_noise-rate-0.2900.pickle'):
    data, label_true, label_noisy = load_data_moon(path)
    data_division_processor = MoonDataSetDivision(data, label_true, label_noisy)
    return data_division_processor


def get_dataset_clusters(clusters=None, path='./dataset_moon/sample-number-500/data_noise-rate-0.2900.pickle'):
    data, label_true, label_noisy = load_data_moon(path)
    dataset = MoonDataSetClusters(data, label_true, label_noisy, clusters)
    return dataset


def generate_dataset_getter(path='./dataset_moon/sample-number-500/data_noise-rate-0.2900.pickle'):
    data, label_true, label_noisy = load_data_moon(path)
    train_data_dict = {
        'data': data,
        'label_true': label_true,
        'label_noisy': label_noisy
    }
    return MoonDataSetGetter(train_data_dict)


class MoonDataSet(Dataset):
    def __init__(self, data, label_true, label_noisy):
        self.data = torch.from_numpy(data).float()
        self.label_true = torch.from_numpy(label_true)
        self.label_noisy = torch.from_numpy(label_noisy)

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        return self.data[item], self.label_true[item], self.label_noisy[item]


class MoonDataSetDivision:
    def __init__(self, data, label_true, label_noisy):
        self.data = data
        self.label_true = label_true
        self.label_noisy = label_noisy

    def __len__(self):
        return len(self.label_true)

    def get_dataset(self, division=None):
        if division is None:
            division = np.ones(len(self), dtype=bool)

        dataset = MoonDataSet(self.data[division],
                              self.label_true[division],
                              self.label_noisy[division]
                              )
        return dataset


class MoonDataSetClusters(Dataset):
    def __init__(self, data, label_true, label_noisy, clusters=None):
        self.clusters = clusters
        self.data = torch.from_numpy(data).float()
        self.label_true = torch.from_numpy(label_true)
        self.label_noisy = torch.from_numpy(label_noisy)

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        return self.data[item], self.label_true[item], self.label_noisy[item], self.clusters[item]

    def set_clusters(self, clusters):
        self.clusters = torch.from_numpy(clusters).long()
        return self


class MoonDataSetDivideMixLabeled(Dataset):
    def __init__(self, data, label_true, label_noisy, prob):
        super(MoonDataSetDivideMixLabeled, self).__init__()
        self.prob = torch.from_numpy(prob).float()
        self.data = torch.from_numpy(data).float()
        self.label_true = torch.from_numpy(label_true)
        self.label_noisy = torch.from_numpy(label_noisy)

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        return self.data[item], self.label_true[item], self.label_noisy[item], self.prob[item]


class MoonDataSetDivideMixUnlabeled(Dataset):
    def __init__(self, data, label_true, label_noisy, prob):
        super(MoonDataSetDivideMixUnlabeled, self).__init__()
        self.prob = torch.from_numpy(prob).float()
        self.data = torch.from_numpy(data).float()
        self.label_true = torch.from_numpy(label_true)
        self.label_noisy = torch.from_numpy(label_noisy)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MoonDataSetGetter:
    def __init__(self, train_data_dict):
        self.train_data_dict = train_data_dict

    def __len__(self):
        return len(self.train_data_dict['label_true'])

    def get_data(self):
        return self.train_data_dict['data']

    @property
    def train_data_base(self):
        return self._get_train_data_base()

    def _get_train_data_base(self):
        data = self.train_data_dict['data']
        label_true = self.train_data_dict['label_true']
        label_noisy = self.train_data_dict['label_noisy']

        return data, label_true, label_noisy

    def get_train_dataset_base(self):
        data, label_true, label_noisy = self._get_train_data_base()
        dataset = MoonDataSet(data, label_true, label_noisy)

        return dataset

    def get_train_dataset_division_for_dividemix(self, prob, pred):
        data, label_true, label_noisy = self._get_train_data_base()
        dataset_labeled = MoonDataSetDivideMixLabeled(data[pred], label_true[pred], label_noisy[pred], prob[pred])
        not_pred = np.logical_not(pred)
        dataset_unlabeled = MoonDataSetDivideMixUnlabeled(data[not_pred], label_true[not_pred],
                                                          label_noisy[not_pred], prob[not_pred])

        return dataset_labeled, dataset_unlabeled
