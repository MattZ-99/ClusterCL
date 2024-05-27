# -*- coding: utf-8 -*-
# @Time : 2022/7/16 19:14
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
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from data.dataProcess import data_class_select
from data.load_data_cifar_100 import *

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

transform_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])
transform_cifar100_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


class ContrastiveLearningViewGenerator:
    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.transform(x) for i in range(self.n_views)]


def cifar10_data_loader_noisy(data_dir='./repositories/data/cifar-10-batches-py',
                              index_root='./repositories/data/cifar-n-processed/cifar-10/2022_7_16_20_32_17',
                              batch_size=1, num_workers=1):
    noise_mode = "worse_label"
    percent = 1
    noise_label_sample_rate = -1
    train_data, train_label, noise_label = data_class_select.load_data(
        data_dir=data_dir,
        index_root=index_root,
        noise_type=noise_mode,
        data_percent=percent,
        noise_rate=noise_label_sample_rate
    )

    train_set = CifarTrainNoisy(train_data, train_label, noise_label, transform=transform_cifar10_train)

    selected_categories = [i for i in range(10)]
    test_data, test_label = data_class_select.load_cifar10_test(
        data_dir,
        selected_categories=selected_categories
    )

    test_set = CifarTest(test_data, test_label, transform=transform_cifar10_test)

    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers)

    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers)

    return train_data_loader, test_data_loader


def generate_cifar_n_dataset_getter(data_dir='./repositories/data/cifar-10-batches-py',
                                    index_root='./repositories/data/cifar-n-processed/cifar-10/2022_7_16_20_32_17',
                                    noise_mode="worse_label", data_percent=1, noise_label_sample_rate=-1
                                    ):
    percent = data_percent
    noise_label_sample_rate = noise_label_sample_rate
    train_data, train_label, noise_label = data_class_select.load_data(
        data_dir=data_dir,
        index_root=index_root,
        noise_type=noise_mode,
        data_percent=percent,
        noise_rate=noise_label_sample_rate
    )
    data_getter = CifarNDatasetGetter(data=train_data, label_true=train_label, label_noise=noise_label)
    return data_getter


class CifarNDatasetGetter:
    def __init__(self, data, label_true, label_noise):
        super(CifarNDatasetGetter, self).__init__()
        self.label_noisy = label_noise
        self.label_true = label_true
        self.data = data

    def get_dataset_base(self, transform=None):
        if transform is None:
            transform = transform_cifar10_train

        return CifarTrainNoisy(self.data, self.label_true, self.label_noisy, transform=transform)

    def get_dataset_eval(self, transform=None):
        if transform is None:
            transform = transform_cifar10_test

        return CifarTrainNoisy(self.data, self.label_true, self.label_noisy, transform=transform)

    def get_dataset_multi_view(self, n_views=2):
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])

        transform = ContrastiveLearningViewGenerator(transform=aug_transform, n_views=n_views)
        return CifarTrainNoisy(self.data, self.label_true, self.label_noisy, transform=transform)

    def get_dataset_noise_only_base(self):
        return CifarTrain(self.data, self.label_noisy, transform=transform_cifar10_train)

    def get_dataset_clean_base(self):
        return CifarTrain(self.data, self.label_true, transform=transform_cifar10_train)

    def get_dataset_cluster(self, clusters):
        return CifarTrainCluster(self.data, self.label_true, self.label_noisy, clusters,
                                 transform=transform_cifar10_train)

    def get_label(self):
        return self.label_true, self.label_noisy

    def get_dataset_corrected(self, label_corrected):
        return CifarTrainCorrected(self.data, self.label_true, label_corrected, transform=transform_cifar10_train)

    @property
    def noise_rate(self):
        return 1 - np.mean(self.label_true == self.label_noisy)


def generate_cifar_n_dataset_getter_dividemix(data_dir='./repositories/data/cifar-10-batches-py',
                                              index_root='./repositories/data/cifar-n-processed/cifar-10/2022_7_16_20_32_17',
                                              noise_mode="worse_label", data_percent=1, noise_label_sample_rate=-1,
                                              class_num=10
                                              ):
    percent = data_percent
    synthetic_noise_flag = False

    if noise_mode == "synthetic_label":
        train_data, train_label, noise_label = data_class_select.load_synthetic_data(
            data_dir=data_dir,
            index_root=index_root,
            data_percent=percent,
            noise_rate=noise_label_sample_rate
        )
    else:
        try:
            train_data, train_label, noise_label = data_class_select.load_data(
                data_dir=data_dir,
                index_root=index_root,
                noise_type=noise_mode,
                data_percent=percent,
                noise_rate=noise_label_sample_rate
            )
        except AssertionError:
            train_data, train_label, noise_label = data_class_select.load_data(
                data_dir=data_dir,
                index_root=index_root,
                noise_type=noise_mode,
                data_percent=percent,
                noise_rate=-1
            )
            synthetic_noise_flag = True

    train_dict = {
        "data": train_data,
        "label_true": train_label,
        "label_noise": noise_label
    }
    if class_num == 10:
        test_data, test_label = data_class_select.load_data_test(path=data_dir)
    elif class_num == 2:
        test_data, test_label = data_class_select.load_data_test(path=data_dir)
        index = (test_label == 3) | (test_label == 5)
        test_data = test_data[index]
        test_label = test_label[index]
        test_label = np.where(test_label == 3, 0, 1)
    else:
        raise NotImplementedError

    test_dict = {
        "data": test_data,
        "label_true": test_label
    }

    data_getter = CifarNDatasetGetterDivideMix(train_dict=train_dict, test_dict=test_dict)
    if synthetic_noise_flag:
        print("Generating synthetic noise...")
        data_getter.generate_synthetic_label_noise(noise_rate=noise_label_sample_rate)
    print(f"Noise rate={data_getter.noise_rate:.4f}")

    return data_getter


def generate_cifar_100n_dataset_getter(data_dir='./repositories/data/cifar-100-python',
                                       noise_path='./data/cifar-n-labels/CIFAR-100_human.pt'
                                       ):
    train_data, train_label, noisy_label = load_train_data_fine(os.path.join(data_dir, 'train'), noise_path)
    train_dict = {
        "data": train_data,
        "label_true": train_label,
        "label_noise": noisy_label
    }
    test_data, test_label = load_test_data_fine(os.path.join(data_dir, 'test'))

    test_dict = {
        "data": test_data,
        "label_true": test_label
    }
    data_getter = CifarNDatasetGetterDivideMix(train_dict=train_dict, test_dict=test_dict,
                                               transform_train=transform_cifar100_train,
                                               transform_test=transform_cifar100_test)
    return data_getter


class CifarNDatasetGetterDivideMix:
    def __init__(self, train_dict, test_dict,
                 transform_train=transform_cifar10_train, transform_test=transform_cifar10_test):
        self.test_dict = test_dict
        self.train_dict = train_dict
        self.transform_test = transform_test
        self.transform_train = transform_train

    def get_dataset_all_labels(self):
        return CifarTrainNoisy(self.train_dict["data"],
                               self.train_dict['label_true'],
                               self.train_dict["label_noise"],
                               transform=self.transform_train)

    def get_warmup_dataset(self):
        return CifarTrain(self.train_dict["data"], self.train_dict["label_noise"], transform=self.transform_train)

    def get_train_clean_dataset(self):
        return CifarTrain(self.train_dict["data"], self.train_dict["label_true"], transform=self.transform_train)

    def get_test_dataset(self):
        return CifarTest(self.test_dict["data"], self.test_dict["label_true"], transform=self.transform_test)

    def get_train_eval_dataset(self, train_percent=0.9):
        num_samples = len(self.train_dict["data"])
        train_set_index = np.random.choice(num_samples, int(num_samples * train_percent), replace=False)
        index = np.arange(num_samples)
        val_set_index = np.delete(index, train_set_index)

        train_dataset = CifarTrain(self.train_dict["data"][train_set_index, :],
                                   self.train_dict["label_noise"][train_set_index],
                                   transform=self.transform_train)
        val_dataset = CifarTrain(self.train_dict["data"][val_set_index, :],
                                 self.train_dict["label_noise"][val_set_index],
                                 transform=self.transform_train)

        return train_dataset, val_dataset

    def get_eval_dataset(self):
        return CifarTrainIndex(self.train_dict["data"], self.train_dict["label_noise"],
                               transform=self.transform_test)

    def get_divide_datasets(self, prob, pred):
        data = self.train_dict["data"]
        label_true = self.train_dict["label_true"]
        label_noise = self.train_dict["label_noise"]

        pred_idx_labeled = pred.nonzero()[0]
        labeled_dataset = CifarTrainLabeled(data[pred_idx_labeled], label_noise[pred_idx_labeled],
                                            prob[pred_idx_labeled], transform=self.transform_train)

        pred_idx_unlabeled = (1 - pred).nonzero()[0]
        unlabeled_dataset = CifarTrainUnlabeled(data=data[pred_idx_unlabeled], transform=self.transform_train)

        return labeled_dataset, unlabeled_dataset

    def get_nmce_dataset(self, n_views=2, contrastive=True, transform=None):
        if transform is None:
            transform = self.transform_train
        if contrastive:
            transform = ContrastiveLearningViewGenerator(transform, n_views=n_views)

        return CifarTrain(self.train_dict["data"], self.train_dict["label_noise"], transform=transform)

    def get_dataset_train_with_data(self, data, labels):
        return CifarTrain(data, labels, transform=self.transform_train)

    @property
    def train_data(self):
        return self.train_dict["data"]

    @property
    def train_labels_noisy(self):
        return self.train_dict["label_noise"]

    @property
    def noise_rate(self, train=True):
        noise_rate = 1 - np.mean(self.train_dict["label_noise"] == self.train_dict["label_true"])

        return noise_rate

    @property
    def train_labels_true(self):
        return self.train_dict["label_true"]

    def set_label_noisy(self, label_noisy):
        self.train_dict["label_noise"] = label_noisy

    def generate_synthetic_label_noise(self, noise_rate=0.2, noise_type='two-classes-symmetric', classes=2, pho_max=1.):
        label_true = self.train_dict["label_true"]
        label_noisy = np.copy(label_true)
        if noise_type == 'two-classes-symmetric':
            index_noisy = np.random.uniform(0, 1, label_true.shape) < noise_rate
            label_noisy = np.where(index_noisy, 1 - label_noisy, label_noisy)
        elif noise_type == 'two-classes-instance-dependent':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.train_dict["label_noise"] = label_noisy

    def get_dataset_for_clr_train_eval(self, train_percent=0.9):
        num_samples = len(self.train_dict["data"])
        train_set_index = np.random.choice(num_samples, int(num_samples * train_percent), replace=False)
        index = np.arange(num_samples)
        val_set_index = np.delete(index, train_set_index)

        train_dataset = CifarForCLR(self.train_dict["data"][train_set_index, :],
                                    label_true=np.array(self.train_dict["label_true"])[train_set_index],
                                    label_noisy=np.array(self.train_dict["label_noise"])[train_set_index],
                                    transform=self.transform_train)
        val_dataset = CifarForCLR(self.train_dict["data"][val_set_index, :],
                                  label_true=np.array(self.train_dict["label_true"])[val_set_index],
                                  label_noisy=np.array(self.train_dict["label_noise"])[val_set_index],
                                  transform=self.transform_test)

        return train_dataset, val_dataset

    def get_dataset_for_sop_train_eval(self, transform_train=None, transform_train_aug=None,
                                       transform_val=None, train_percent=0.9):
        num_samples = len(self.train_dict["data"])
        train_set_index = np.random.choice(num_samples, int(num_samples * train_percent), replace=False)
        index = np.arange(num_samples)
        val_set_index = np.delete(index, train_set_index)

        train_dataset = CifarForSOP(self.train_dict["data"][train_set_index, :],
                                    label_true=np.array(self.train_dict["label_true"])[train_set_index],
                                    label_noisy=np.array(self.train_dict["label_noise"])[train_set_index],
                                    transform=transform_train, transform_aug=transform_train_aug)
        val_dataset = CifarForCLR(self.train_dict["data"][val_set_index, :],
                                  label_true=np.array(self.train_dict["label_true"])[val_set_index],
                                  label_noisy=np.array(self.train_dict["label_noise"])[val_set_index],
                                  transform=transform_val)

        return train_dataset, val_dataset

    def get_dataset_for_clr_test(self, transform=None):
        if transform is None:
            transform = self.transform_test
        test_dataset = CifarForCLR(self.test_dict["data"],
                                   label_true=self.test_dict["label_true"],
                                   label_noisy=self.test_dict["label_true"],
                                   transform=transform)
        return test_dataset


class CifarTrainLabeled(Dataset):
    def __init__(self, data, label, prob, transform):
        super(CifarTrainLabeled, self).__init__()
        self.prob = prob
        self.label = label
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img, target, prob = self.data[item], self.label[item], self.prob[item]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2, target, prob


class CifarTrainUnlabeled(Dataset):
    def __init__(self, data, transform):
        super(CifarTrainUnlabeled, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2


class CifarTrainCorrected(Dataset):
    def __init__(self, data, label_true, label_corrected, transform):
        super(CifarTrainCorrected, self).__init__()
        self.label_corrected = label_corrected
        self.label_true = label_true
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label_true[item], self.label_corrected[item]

    @property
    def noise_rate(self) -> float:
        noise_rate = 1 - np.mean(self.label_corrected == self.label_true)

        return noise_rate


class CifarTrainCluster(Dataset):
    def __init__(self, data, label_true, label_noisy, cluster, transform):
        super(CifarTrainCluster, self).__init__()
        self.data = data
        self.label_true = label_true
        self.label_noisy = label_noisy
        self.cluster = torch.from_numpy(cluster).long()
        self.transform = transform

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label_true[item], self.label_noisy[item], self.cluster[item]


class CifarTrainNoisy(Dataset):
    def __init__(self, data, label_true, label_noisy, transform):
        super(CifarTrainNoisy, self).__init__()
        self.data = data
        self.label_true = label_true
        self.label_noisy = label_noisy
        self.transform = transform

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label_true[item], self.label_noisy[item]


class CifarTrainIndex(Dataset):
    def __init__(self, data, label, transform):
        super(CifarTrainIndex, self).__init__()
        self.label = label
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label[item], item


class CifarTrain(Dataset):
    def __init__(self, data, label, transform):
        super(CifarTrain, self).__init__()
        self.label = label
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label[item]


class CifarTest(Dataset):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label[item]


class CifarForCLR(Dataset):
    def __init__(self, data, label_true, label_noisy, transform):
        super(CifarForCLR, self).__init__()
        self.label_noisy = label_noisy
        self.label_true = label_true
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.label_noisy[item], item, self.label_true[item]


class CifarForSOP(Dataset):
    def __init__(self, data, label_true, label_noisy, transform, transform_aug):
        super(CifarForSOP, self).__init__()
        self.transform_aug = transform_aug
        self.label_noisy = label_noisy
        self.label_true = label_true
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.label_true)

    def __getitem__(self, item):
        img = self.data[item]
        img = Image.fromarray(img)
        return self.transform(img), self.transform_aug(img), self.label_noisy[item], item, self.label_true[item]