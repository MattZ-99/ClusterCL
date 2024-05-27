# -*- coding: utf-8 -*-
# @Time : 2022/10/13 14:58
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
from torchvision.transforms import transforms
from . import dataset_cifar
from .arguments.gaussian_blur import GaussianBlur
from .arguments.view_generator import ContrastiveLearningViewGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from data.dataProcess import data_class_select


# ------------------------------------------------------------------------------
# Get generators.

## get CIFAR10N dataset generator.
def get_cifar10n_dataset_generator(data_dir='./data/cifar-10-batches-py',
                                   index_root='./data/cifar-n-processed/cifar-10/2022_9_3_10_51_27',
                                   noise_mode='worse_label', data_percent=1, noise_rate=-1, *args, **kwargs):
    # Load data.
    train_data, train_label, noise_label = data_class_select.load_data(
        data_dir=data_dir,
        index_root=index_root,
        noise_type=noise_mode,
        data_percent=data_percent,
        noise_rate=noise_rate, *args, **kwargs
    )

    test_data, test_label = data_class_select.load_data_test(path=data_dir)

    # Dict format for dataset generator.
    train_dict = {
        "data": train_data,
        "label_true": train_label,
        "label_noise": noise_label
    }

    test_dict = {
        "data": test_data,
        "label_true": test_label
    }

    transform_dict = {
        'transform_cifar10_train': dataset_cifar.transform_cifar10_train,
        'transform_cifar10_test': dataset_cifar.transform_cifar10_test
    }

    dataset_generator = CIFAR10NDatasetGenerator(train_dict, test_dict, transform_dict)

    return dataset_generator


# ------------------------------------------------------------------------------
# Dataset generators.

## Base dataset generator.
class BaseDatasetGenerator:
    """ Base dataset generator.
    train_dict: dict, train data dict.
                 train_dict = {
                    "data": train_data,
                    "label_true": train_label,
                    "label_noise": noise_label
                }

    test_dict: dict, test data dict.
                test_dict = {
                    "data": test_data,
                    "label_true": test_label
                }

    transform_dict: dict, transform dict.

    """

    def __init__(self, train_dict, test_dict, transform_dict):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.transform_dict = transform_dict

    @property
    def train_data(self):
        return self.train_dict["data"]

    @property
    def train_label_true(self):
        return self.train_dict["label_true"]

    @property
    def train_label_noise(self):
        return self.train_dict["label_noise"]

    @property
    def train_label_clean_or_not(self):
        return self.train_label_noise == self.train_label_true

    @property
    def train_label_noise_rate(self):
        return np.sum(self.train_label_noise != self.train_label_true) / len(self.train_label_true)

    @property
    def test_data(self):
        return self.test_dict["data"]

    @property
    def test_label_true(self):
        return self.test_dict["label_true"]


## CIFAR10N dataset generator.
class CIFAR10NDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, train_dict, test_dict, transform_dict):
        super().__init__(train_dict, test_dict, transform_dict)

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def generate_train_clean_dataset(self):
        return dataset_cifar.SimpleDataset(self.train_data,
                                           self.train_label_true,
                                           self.transform_dict["transform_cifar10_train"]
                                           )

    def generate_train_noisy_dataset(self):
        return dataset_cifar.SimpleDataset(self.train_data,
                                           self.train_label_noise,
                                           self.transform_dict["transform_cifar10_train"]
                                           )

    def generate_train_simclr_dataset(self, n_views=2):
        return dataset_cifar.UnsupervisedDataset(self.train_data,
                                                 ContrastiveLearningViewGenerator(
                                                     self.get_simclr_pipeline_transform(32), n_views)
                                                 )

    def generate_eval_noisy_dataset(self):
        return dataset_cifar.SimpleDataset(self.train_data,
                                           self.train_label_noise,
                                           self.transform_dict["transform_cifar10_test"]
                                           )

    def generate_test_dataset(self):
        return dataset_cifar.SimpleDataset(self.test_data,
                                           self.test_label_true,
                                           self.transform_dict["transform_cifar10_test"]
                                           )

    def generate_train_clean_noisy_dataset(self, transform=None):
        if transform is None:
            transform = self.transform_dict["transform_cifar10_train"]
        elif isinstance(transform, str):
            if transform == "train":
                transform = self.transform_dict["transform_cifar10_train"]
            elif transform == "test":
                transform = self.transform_dict["transform_cifar10_test"]
            else:
                raise ValueError("transform must be 'train' or 'test'.")

        return dataset_cifar.LabelTrueAndNoisyDataset(self.train_data,
                                                      self.train_label_true,
                                                      self.train_label_noise,
                                                      transform
                                                      )

    def generate_train_clean_weak_strong_dataset(self):
        weak_argument = self.transform_dict["transform_cifar10_train"]
        strong_argument = self.transform_dict.setdefault("transform_cifar10_train_strong",
                                                         dataset_cifar.transform_cifar10_train_randaugment)
        return dataset_cifar.MultiAugDataset(self.train_data,
                                             self.train_label_true,
                                             [weak_argument, strong_argument]
                                             )

    def generate_train_noisy_weak_strong_dataset(self):
        weak_argument = self.transform_dict["transform_cifar10_train"]
        strong_argument = self.transform_dict.setdefault("transform_cifar10_train_strong",
                                                         dataset_cifar.transform_cifar10_train_randaugment)
        return dataset_cifar.MultiAugDataset(self.train_data,
                                             self.train_label_noise,
                                             [weak_argument, strong_argument]
                                             )

    def generate_train_noisy_weak_strong_dataset_with_preds(self, pred1, pred2):
        weak_argument = self.transform_dict["transform_cifar10_train"]
        strong_argument = self.transform_dict.setdefault("transform_cifar10_train_strong",
                                                         dataset_cifar.transform_cifar10_train_randaugment)

        return dataset_cifar.MultiAugDatasetWithPreds(self.train_data,
                                                      self.train_label_noise, pred1, pred2,
                                                      [weak_argument, strong_argument])

    def generate_train_noisy_divided_datasets(self, prob, pred):
        weak_argument = self.transform_dict["transform_cifar10_train"]
        strong_argument = self.transform_dict.setdefault("transform_cifar10_train_strong",
                                                         dataset_cifar.transform_cifar10_train_randaugment)

        data = self.train_data
        label_true = self.train_label_true
        label_noise = self.train_label_noise

        pred_idx_labeled = pred.nonzero()[0]
        labeled_dataset = dataset_cifar.LabeledDatasetWithProbAndMultiAug(
            data[pred_idx_labeled], label_noise[pred_idx_labeled],
            prob[pred_idx_labeled], transform=[weak_argument, strong_argument]
        )

        pred_idx_unlabeled = (1 - pred).nonzero()[0]
        unlabeled_dataset = dataset_cifar.UnlabeledDatasetWithMultiAug(
            data=data[pred_idx_unlabeled], transform=[weak_argument, weak_argument]
        )

        return labeled_dataset, unlabeled_dataset

    def generate_merged_labeled_and_unlabeled_dataset(self, label_labeled, label_unlabeled, label_pred):
        weak_argument = self.transform_dict["transform_cifar10_train"]
        data = self.train_data

        labels = np.empty(shape=(label_pred.shape[0], label_labeled.shape[1]))
        pred_idx_labeled = label_pred.nonzero()[0]
        labels[pred_idx_labeled] = label_labeled
        pred_idx_unlabeled = (1 - label_pred).nonzero()[0]
        labels[pred_idx_unlabeled] = label_unlabeled

        labeled_dataset = dataset_cifar.LabeledDatasetWithProbAndMultiAug(
            data, labels, label_pred, transform=weak_argument
        )

        return labeled_dataset
