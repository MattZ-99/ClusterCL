# -*- coding: utf-8 -*-
# @Time : 2022/7/15 19:59
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
import torch

transform_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


def cifar10_data_loader(root='./repositories/data', batch_size=1, num_workers=1):
    train_data = datasets.CIFAR10(root=root, train=True, transform=transform_cifar10_train)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers)

    test_data = datasets.CIFAR10(root=root, train=False, transform=transform_cifar10_test)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)

    return train_data_loader, test_data_loader
