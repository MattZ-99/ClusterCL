# -*- coding: utf-8 -*-
# @Time : 2022/7/17 20:22
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

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from tools import utils


def plot_data(data, label, ax, num_color=2, show_legend=True):
    for i in range(num_color):
        ax.scatter(*data[label == i].T, s=4, alpha=0.5, label=f'Class {i}')
    if show_legend:
        ax.legend()


def generate_noisy_label_cluster_noise(data, label_true, label_clusters, n_clusters=10):
    true_label = []
    data_x_mean = []
    for i in range(n_clusters):
        if np.mean(label_true[label_clusters == i]) > 0.5:
            true_label.append(1)
        else:
            true_label.append(0)

        data_x_mean.append(np.mean(data[:, 0][label_clusters == i]))

    true_label = np.array(true_label)
    data_x_mean = np.array(data_x_mean)

    arg_value = np.argsort(data_x_mean)

    # print(true_label)
    # print(data_x_mean)
    # print(arg_value)

    flip_rate = np.zeros(n_clusters)

    noise_rate_0 = np.linspace(0.05, 0.45, (true_label == 0).sum())
    noise_rate_1 = np.linspace(0.45, 0.05, (true_label == 0).sum())

    c_0 = 0
    c_1 = 0
    for i in range(n_clusters):
        index = arg_value[i]
        if true_label[index] == 0:
            flip_rate[index] = noise_rate_0[c_0]
            c_0 += 1
        elif true_label[index] == 1:
            flip_rate[index] = noise_rate_1[c_1]
            c_1 += 1

    instance_flip_rate = np.zeros(len(data))
    for i in range(n_clusters):
        instance_flip_rate[label_clusters == i] = flip_rate[i]

    p = np.random.uniform(low=0, high=1, size=len(data))

    label_noisy = np.where(p < instance_flip_rate, 1 - label_true, label_true)

    print(flip_rate[true_label == 0], flip_rate[true_label == 1])
    print(f"noisy rate: {1 - np.mean(label_noisy == label_true)}")
    index = (label_true == 0)
    print(f"noisy rate (true class 0): {1 - np.mean(label_noisy[index] == label_true[index])}")
    index = (label_true == 1)
    print(f"noisy rate (true class 1): {1 - np.mean(label_noisy[index] == label_true[index])}")

    return label_noisy


def generate_noisy_label_reverse_cluster_noise(data, label_true, label_clusters):
    true_label = []
    data_x_mean = []
    for i in range(10):
        if np.mean(label_true[label_clusters == i]) > 0.5:
            true_label.append(1)
        else:
            true_label.append(0)

        data_x_mean.append(np.mean(data[:, 0][label_clusters == i]))

    true_label = np.array(true_label)
    data_x_mean = np.array(data_x_mean)

    arg_value = np.argsort(data_x_mean)

    print(true_label)
    print(data_x_mean)
    print(arg_value)

    flip_rate = np.zeros(10)
    noise_rate_0 = [0.45, 0.35, 0.25, 0.15, 0.05]
    noise_rate_1 = [0.05, 0.15, 0.25, 0.35, 0.45]

    c_0 = 0
    c_1 = 0
    for i in range(10):
        index = arg_value[i]
        if true_label[index] == 0:
            flip_rate[index] = noise_rate_0[c_0]
            c_0 += 1
        elif true_label[index] == 1:
            flip_rate[index] = noise_rate_1[c_1]
            c_1 += 1

    instance_flip_rate = np.zeros(len(data))
    for i in range(10):
        instance_flip_rate[label_clusters == i] = flip_rate[i]

    p = np.random.uniform(low=0, high=1, size=len(data))

    label_noisy = np.where(p < instance_flip_rate, 1 - label_true, label_true)

    print(f"noisy rate: {1 - np.mean(label_noisy == label_true)}")

    return label_noisy


def generate_noisy_label_uniform_noise(data, label_true, noise_rate=0.1):
    p = np.random.uniform(low=0, high=1, size=len(data))
    label_noisy = np.where(p < noise_rate, 1 - label_true, label_true)

    return label_noisy


def generate_moon_data_cluster_noise(num_samples=100, path='./'):
    path = os.path.join(path, 'noise-type-clusters', f'sample-number-{num_samples}')
    utils.makedirs(path)

    data, label_true = datasets.make_moons(n_samples=num_samples, noise=0.05)

    n_clusters = 30
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    label_clusters = kmeans.fit_predict(data)

    label_noisy = generate_noisy_label_cluster_noise(data, label_true, label_clusters, n_clusters=n_clusters)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plot_data(data, label_true, ax=ax[0])
    plot_data(data, label_clusters, ax=ax[1], num_color=n_clusters, show_legend=False)
    plot_data(data, label_noisy, ax=ax[2])
    plt.savefig(os.path.join(path, './dataset_moon.png'))

    moon_data_dict = {
        'data': data,
        'label_true': label_true,
        'label_noisy': label_noisy
    }

    utils.save_pickle(moon_data_dict,
                      os.path.join(path, f'data_noise-rate-{1 - np.mean(label_noisy == label_true):.4f}.pickle'))


def generate_moon_data_reverse_cluster_noise(num_samples=100, path='./'):
    path = os.path.join(path, 'noise-type-reverse-clusters', f'sample-number-{num_samples}')
    utils.makedirs(path)

    data, label_true = datasets.make_moons(n_samples=num_samples, noise=0.05)

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    label_clusters = kmeans.fit_predict(data)

    label_noisy = generate_noisy_label_reverse_cluster_noise(data, label_true, label_clusters)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plot_data(data, label_true, ax=ax[0])
    plot_data(data, label_clusters, ax=ax[1], num_color=n_clusters)
    plot_data(data, label_noisy, ax=ax[2])
    plt.savefig(os.path.join(path, './dataset_moon.png'))

    moon_data_dict = {
        'data': data,
        'label_true': label_true,
        'label_noisy': label_noisy
    }

    utils.save_pickle(moon_data_dict,
                      os.path.join(path, f'data_noise-rate-{1 - np.mean(label_noisy == label_true):.4f}.pickle'))


def generate_moon_data_with_uniform_noise(num_samples=100, path='./', noise_rate=0.1):
    path = os.path.join(path, 'noise-type-uniform', f'noise-rate-{noise_rate:.2f}_sample-number-{num_samples}')
    utils.makedirs(path)

    data, label_true = datasets.make_moons(n_samples=num_samples, noise=0.05)
    label_noisy = generate_noisy_label_uniform_noise(data, label_true, noise_rate=noise_rate)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plot_data(data, label_true, ax=ax[0])
    plot_data(data, label_noisy, ax=ax[1])
    plt.savefig(os.path.join(path, './dataset_moon.png'))

    moon_data_dict = {
        'data': data,
        'label_true': label_true,
        'label_noisy': label_noisy
    }

    utils.save_pickle(moon_data_dict,
                      os.path.join(path, f'data_noise-rate-{1 - np.mean(label_noisy == label_true):.4f}.pickle'))


if __name__ == '__main__':
    generate_moon_data_cluster_noise(num_samples=2000, path='./dataset_moon/')
    # generate_moon_data_with_uniform_noise(path='./dataset_moon/', num_samples=5000, noise_rate=0.2)
    # generate_moon_data_reverse_cluster_noise(num_samples=2000, path='./dataset_moon/')
