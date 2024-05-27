# -*- coding: utf-8 -*-
# @Time : 2022/10/13 15:24
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

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from .arguments.randaug import RandomAugment

# ------------------------------------------------------------------------------
# Transforms.

## CIFAR10 base train transform.
transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

## CIFAR10 base test transform.
transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

## CIFAR100 base train transform.
transform_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

## CIFAR100 base test transform.
transform_cifar100_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

## CIFAR10 Random RandAugment transform.
transform_cifar10_train_randaugment = transforms.Compose([
    RandomAugment(3, 5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


# ------------------------------------------------------------------------------
# Datasets.

## Simple dataset.
class SimpleDataset(Dataset):
    def __init__(self, data, label, transform=None):
        super(SimpleDataset, self).__init__()
        self.transform = transform
        self.label = label
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.label[item]


## Both true and noisy label dataset.
class LabelTrueAndNoisyDataset(Dataset):
    def __init__(self, data, label_ture, label_noisy, transform=None):
        super(LabelTrueAndNoisyDataset, self).__init__()
        self.transform = transform
        self.label_true = label_ture
        self.label_noisy = label_noisy
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.label_true[item], self.label_noisy[item]


## Multiple augmentations dataset.
class MultiAugDataset(Dataset):
    def __init__(self, data, label, transform=None):
        super(MultiAugDataset, self).__init__()
        self.label = label
        self.data = data

        if transform is not None:
            if not isinstance(transform, list):
                transform = [transform]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)

        img_list = []
        if self.transform is None:
            img_list.append(img)
        else:
            for t in self.transform:
                img_list.append(t(img))

        return img_list, self.label[item]


## Multiple augmentations dataset with preds.
class MultiAugDatasetWithPreds(Dataset):
    def __init__(self, data, label, pred1, pred2, transform=None):
        super(MultiAugDatasetWithPreds, self).__init__()

        self.pred2 = pred2
        self.pred1 = pred1

        self.label = label
        self.data = data

        if transform is not None:
            if not isinstance(transform, list):
                transform = [transform]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)

        img_list = []
        if self.transform is None:
            img_list.append(img)
        else:
            for t in self.transform:
                img_list.append(t(img))

        return img_list, self.label[item], self.pred1[item], self.pred2[item]


## Labeled dataset with prob.
class LabeledDatasetWithProbAndMultiAug(Dataset):
    def __init__(self, data, label, prob, transform=None):
        super(LabeledDatasetWithProbAndMultiAug, self).__init__()
        self.prob = prob
        self.label = label
        self.data = data

        if transform is not None:
            if not isinstance(transform, list):
                transform = [transform]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target, prob = self.data[item], self.label[item], self.prob[item]
        img = Image.fromarray(img)

        img_list = []
        if self.transform is None:
            img_list.append(img)
        else:
            for t in self.transform:
                img_list.append(t(img))

        return img_list, target, prob


## Unlabeled dataset with multiple augmentations.
class UnlabeledDatasetWithMultiAug(Dataset):
    def __init__(self, data, transform=None):
        super(UnlabeledDatasetWithMultiAug, self).__init__()
        self.data = data

        if transform is not None:
            if not isinstance(transform, list):
                transform = [transform]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)

        img_list = []
        if self.transform is None:
            img_list.append(img)
        else:
            for t in self.transform:
                img_list.append(t(img))

        return img_list


## Unsupervised dataset.
class UnsupervisedDataset(Dataset):
    def __init__(self, data, transform=None):
        super(UnsupervisedDataset, self).__init__()
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img
