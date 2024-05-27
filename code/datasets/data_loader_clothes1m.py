# -*- coding: utf-8 -*-
# @Time : 2022/9/25 8:05
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
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch import nn
from data.Clothes1M import load_data


def generate_clothes1m_dataset_getter(root_dir='./data/Clothes1M', num_samples=-1):
    dataset_getter = DatasetGetterClothes1M(root_dir=root_dir, num_samples=num_samples)

    return dataset_getter


class DatasetGetterClothes1M:
    def __init__(self, root_dir='./data/Clothes1M', num_samples=-1):
        self.train_data, self.noisy_labels, self.test_data, self.clean_labels = load_data.load_data(root_dir)
        self.train_data = self.select_samples(self.train_data, num_samples)

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

        self.root_dir = root_dir

    @staticmethod
    def select_samples(train_data, num_samples):
        if num_samples == -1:
            return train_data
        else:
            random.seed(num_samples)
            return random.sample(train_data, num_samples)

    def get_warmup_dataset(self, num_samples: int = 1000):
        train_data = self.get_random_sample_for_data(self.train_data, num_samples)
        return DatasetClothes1M(train_data, self.noisy_labels, self.root_dir, self.transform_train)

    def get_test_dataset(self):
        return DatasetClothes1M(self.test_data, self.clean_labels, self.root_dir, self.transform_test)

    def get_eval_dataset(self, num_samples: int = 1000):
        train_data = self.get_random_sample_for_data(self.train_data, num_samples)
        return DatasetClothes1MIndex(train_data, self.noisy_labels, self.root_dir, self.transform_test)

    def get_train_eval_dataset(self, train_percent=0.9):
        num_samples = len(self.train_data)
        train_set_index = np.random.choice(num_samples, int(num_samples * train_percent), replace=False)
        index = np.arange(num_samples)
        val_set_index = np.delete(index, train_set_index)

        train_part = [self.train_data[i] for i in train_set_index]

        train_dataset = DatasetClothes1M(train_part,
                                         self.noisy_labels, self.root_dir,
                                         transform=self.transform_train)
        val_part = [self.train_data[i] for i in val_set_index]
        val_dataset = DatasetClothes1M(val_part,
                                       self.noisy_labels, self.root_dir,
                                       transform=self.transform_train)

        return train_dataset, val_dataset

    def get_dataset_for_sop_train_eval(self, transform_train=None, transform_train_aug=None,
                                       transform_val=None, train_percent=0.9):
        num_samples = len(self.train_data)
        train_set_index = np.random.choice(num_samples, int(num_samples * train_percent), replace=False)
        index = np.arange(num_samples)
        val_set_index = np.delete(index, train_set_index)

        train_part = [self.train_data[i] for i in train_set_index]
        train_dataset = DatasetClothes1MForSOP(train_part,
                                               self.noisy_labels, self.root_dir,
                                               transform_train, transform_train_aug)
        val_part = [self.train_data[i] for i in val_set_index]
        val_dataset = DatasetClothes1MForCLR(val_part,
                                             self.noisy_labels, self.root_dir,
                                             transform_val)

        return train_dataset, val_dataset

    def get_divide_datasets(self, data, pred, prob):
        noisy_labels = self.noisy_labels
        pred_idx_labeled = pred.nonzero()[0]
        data_labeled = [data[i] for i in pred_idx_labeled]
        labeled_dataset = DatasetClothes1MLabeled(data_labeled, noisy_labels, prob[pred_idx_labeled],
                                                  self.root_dir, transform=self.transform_train)

        pred_idx_unlabeled = (1 - pred).nonzero()[0]
        data_unlabeled = [data[i] for i in pred_idx_unlabeled]
        unlabeled_dataset = DatasetClothes1MUnlabeled(data_unlabeled, self.root_dir, transform=self.transform_train)

        return labeled_dataset, unlabeled_dataset

    @staticmethod
    def get_random_sample_for_data(data, num_samples):
        num_samples = min(len(data), num_samples)
        return random.sample(data, num_samples)

    def get_dataset_for_clr_train_eval(self, train_percent=0.9):
        num_samples = len(self.train_data)
        train_set_index = np.random.choice(num_samples, int(num_samples * train_percent), replace=False)
        index = np.arange(num_samples)
        val_set_index = np.delete(index, train_set_index)

        train_part = [self.train_data[i] for i in train_set_index]

        train_dataset = DatasetClothes1MForCLR(train_part,
                                               self.noisy_labels, self.root_dir,
                                               transform=self.transform_train)

        val_part = [self.train_data[i] for i in val_set_index]
        val_dataset = DatasetClothes1MForCLR(val_part,
                                             self.noisy_labels, self.root_dir,
                                             transform=self.transform_train)

        return train_dataset, val_dataset

    def get_dataset_for_clr_test(self, transform=None):
        if transform is None:
            transform = self.transform_test
        test_dataset = DatasetClothes1MForCLR(self.test_data, self.clean_labels, self.root_dir, transform=transform)
        return test_dataset


class DatasetClothes1M(nn.Module):
    def __init__(self, data, label, root_dir, transform=None):
        super(DatasetClothes1M, self).__init__()
        self.root_dir = root_dir
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.data[idx])).convert('RGB')
        label = self.label[self.data[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


class DatasetClothes1MIndex(nn.Module):
    def __init__(self, data, label, root_dir, transform=None):
        super(DatasetClothes1MIndex, self).__init__()
        self.root_dir = root_dir
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.data[idx])).convert('RGB')
        label = self.label[self.data[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label, idx


class DatasetClothes1MLabeled(nn.Module):
    def __init__(self, data, label, prob, root_dir, transform=None):
        super(DatasetClothes1MLabeled, self).__init__()

        self.root_dir = root_dir
        self.prob = prob
        self.label = label
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.data[idx])).convert('RGB')
        label = self.label[self.data[idx]]
        prob = self.prob[idx]

        img1 = self.transform(image)
        img2 = self.transform(image)
        return img1, img2, label, prob


class DatasetClothes1MUnlabeled(nn.Module):
    def __init__(self, data, root_dir, transform=None):
        super(DatasetClothes1MUnlabeled, self).__init__()

        self.root_dir = root_dir
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.data[idx])).convert('RGB')

        img1 = self.transform(image)
        img2 = self.transform(image)
        return img1, img2


class DatasetClothes1MForCLR(nn.Module):
    def __init__(self, data, label, root_dir, transform=None):
        super(DatasetClothes1MForCLR, self).__init__()
        self.root_dir = root_dir
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.data[idx])).convert('RGB')
        label = self.label[self.data[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label, idx, -1


class DatasetClothes1MForSOP(nn.Module):
    def __init__(self, data, label, root_dir, transform=None, transform_aug=None):
        super(DatasetClothes1MForSOP, self).__init__()
        self.transform_aug = transform_aug
        self.root_dir = root_dir
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.data[idx])).convert('RGB')
        label = self.label[self.data[idx]]

        return self.transform(image), self.transform_aug(image), label, idx, -1
