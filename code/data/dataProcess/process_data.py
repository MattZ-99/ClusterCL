# -*- coding: utf-8 -*-
# @Time : 2022/11/12 13:30
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
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import tools.utils as utils
from nets.wrapper_net import get_single_network
from datasets.dataset_cifar import SimpleDataset
from Modules.tools.cluster_funcs import cluster_kmeans

if __name__ == "__main__":
    import data_divide
else:
    from . import data_divide


def run_generate_synthetic_label_noise():
    output_dir = f"../cifar-synthetic-processed/cifar-10/{utils.get_timestamp()}"
    # generate_asymmetric_label_noise(output_dir=output_dir)
    # generate_symmetric_label_noise(output_dir=output_dir)
    # generate_cluster_dependent_label_noise(output_dir=output_dir)
    generate_class_dependent_label_noise(output_dir=output_dir)


# ------------------------------------------------------------------------------
# Generate label noise and perform sample selection.
## Generate synthetic symmetric label noise.
def generate_symmetric_label_noise(data_path="../cifar-10-batches-py",
                                   output_dir="../cifar-synthetic-processed/cifar-10/",
                                   data_percent_list=None, noise_rate_list=None):
    if data_percent_list is None:
        data_percent_list = [0.002 * (i + 1) for i in range(4)] + \
                            [0.01 * (i + 1) for i in range(9)] + \
                            [0.1 * (i + 1) for i in range(10)]

    if noise_rate_list is None:
        max_noise_rate = 1
        noise_rate_list = [round(0.01 * i, 5) for i in range(int(max_noise_rate * 100) + 1)]

    output_dir = os.path.join(output_dir, 'symmetric_label')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    utils.makedirs(output_dir)

    data, label = data_divide.load_cifar10(data_path)
    label = np.array(label, dtype=int)

    for data_percent in data_percent_list:
        output_dir_data_percent = os.path.join(output_dir, f"data-sample-ratio-{data_percent:.4f}")
        utils.makedirs(output_dir_data_percent)

        index = generate_random_index(total_num=data.shape[0], choice_num=data.shape[0] * data_percent)
        label_true = label[index]

        np.save(os.path.join(output_dir_data_percent, 'data-index.npy'), index)
        np.save(os.path.join(output_dir_data_percent, 'clean-label.npy'), label_true)

        # ------------------------------------------------------------------------------
        Output_str = f"Dataset Size:\t{len(index)}" + '\n'
        categories = np.unique(label_true)
        Output_str += f"Categories number: {len(categories)}" + '\n'
        transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
        for i in range(len(label_true)):
            transition_matrix[label_true[i]][label_true[i]] += 1
        info_file = open(os.path.join(output_dir_data_percent, 'info.txt'), 'w')
        info_file.write('-' * 10 + '\n')
        info_file.write(f"Original label matrix (noise rate={0:.4f}):\n")
        np.savetxt(info_file, X=transition_matrix, fmt="%i")
        info_file.write('-' * 10 + '\n')
        # ------------------------------------------------------------------------------

        for noise_rate in noise_rate_list:
            print(f"Processing| Data percent: {data_percent:.4f}, Noise rate: {noise_rate:.4f}")

            label_noisy = generate_symmetric_noisy_label_cifar10(len(index), label=label_true, noise_rate=noise_rate)
            np.save(os.path.join(output_dir_data_percent, f"noisy-rate-{noise_rate:.4f}.npy"), label_noisy)

            # ------------------------------------------------------------------------------
            info_file.write(f"Modified label matrix (noise rate={noise_rate:.4f}):\n")
            modified_label = label_noisy

            transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
            for i in range(len(label_true)):
                transition_matrix[label_true[i]][modified_label[i]] += 1
            np.savetxt(info_file, X=transition_matrix, fmt="%i")

            info_file.write('-' * 10 + '\n')
            # ------------------------------------------------------------------------------
        info_file.close()


## Generate synthetic asymmetric label noise.
def generate_asymmetric_label_noise(data_path="../cifar-10-batches-py",
                                    output_dir="../cifar-synthetic-processed/cifar-10/",
                                    data_percent_list=None, noise_rate_list=None):
    if data_percent_list is None:
        data_percent_list = [0.002 * (i + 1) for i in range(4)] + \
                            [0.01 * (i + 1) for i in range(9)] + \
                            [0.1 * (i + 1) for i in range(10)]

    if noise_rate_list is None:
        max_noise_rate = 1
        noise_rate_list = [round(0.01 * i, 5) for i in range(int(max_noise_rate * 100) + 1)]

    output_dir = os.path.join(output_dir, 'asymmetric_label')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    utils.makedirs(output_dir)

    data, label = data_divide.load_cifar10(data_path)
    label = np.array(label, dtype=int)

    for data_percent in data_percent_list:
        output_dir_data_percent = os.path.join(output_dir, f"data-sample-ratio-{data_percent:.4f}")
        utils.makedirs(output_dir_data_percent)

        index = generate_random_index(total_num=data.shape[0], choice_num=data.shape[0] * data_percent)
        label_true = label[index]

        np.save(os.path.join(output_dir_data_percent, 'data-index.npy'), index)
        np.save(os.path.join(output_dir_data_percent, 'clean-label.npy'), label_true)

        # ------------------------------------------------------------------------------
        Output_str = f"Dataset Size:\t{len(index)}" + '\n'
        categories = np.unique(label_true)
        Output_str += f"Categories number: {len(categories)}" + '\n'
        transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
        for i in range(len(label_true)):
            transition_matrix[label_true[i]][label_true[i]] += 1
        info_file = open(os.path.join(output_dir_data_percent, 'info.txt'), 'w')
        info_file.write('-' * 10 + '\n')
        info_file.write(f"Original label matrix (noise rate={0:.4f}):\n")
        np.savetxt(info_file, X=transition_matrix, fmt="%i")
        info_file.write('-' * 10 + '\n')
        # ------------------------------------------------------------------------------

        for noise_rate in noise_rate_list:
            print(f"Processing| Data percent: {data_percent:.4f}, Noise rate: {noise_rate:.4f}")

            label_noisy = generate_asymmetric_noisy_label_cifar10(len(index), label=label_true, noise_rate=noise_rate)
            np.save(os.path.join(output_dir_data_percent, f"noisy-rate-{noise_rate:.4f}.npy"), label_noisy)

            # ------------------------------------------------------------------------------
            info_file.write(f"Modified label matrix (noise rate={noise_rate:.4f}):\n")
            modified_label = label_noisy

            transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
            for i in range(len(label_true)):
                transition_matrix[label_true[i]][modified_label[i]] += 1
            np.savetxt(info_file, X=transition_matrix, fmt="%i")

            info_file.write('-' * 10 + '\n')
            # ------------------------------------------------------------------------------
        info_file.close()


## Generate synthetic cluster dependent label noise.
def generate_cluster_dependent_label_noise(data_path="../cifar-10-batches-py",
                                           output_dir="../cifar-synthetic-processed/cifar-10/",
                                           data_percent_list=None, noise_rate_list=None):
    if data_percent_list is None:
        # data_percent_list = [0.002 * (i + 1) for i in range(4)] + \
        #                     [0.01 * (i + 1) for i in range(9)] + \
        #                     [0.1 * (i + 1) for i in range(10)]
        data_percent_list = [1.]

    cluster_num_list = [int(i * 0.01 * 50000) for i in data_percent_list]

    average_noise_rate = 0.4
    cluster_noise_type = "uniform"

    if noise_rate_list is None:
        max_noise_rate = average_noise_rate
        noise_rate_list = [round(0.01 * i, 5) for i in range(int(max_noise_rate * 100) + 1)]

    cluster_noise_rate_left_range = noise_rate_list

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(output_dir, 'cluster_dependent_label')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    utils.makedirs(output_dir)

    data, label = data_divide.load_cifar10(data_path)
    label = np.array(label, dtype=int)

    # Load Clean Net.
    net = get_single_network('pre_resnet18', 10).to(device)
    checkpoint = torch.load(
        '../../Outputs/2022/11/13/cifar10/single_network-TrainCleanMixupCR/class-num-10/worse_label/data-1.00_noise--1.00_pre_resnet18_epochs-100_seed-0_2022_11_13_11_24_3/models/epoch_99.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    for idx, data_percent in enumerate(data_percent_list):
        output_dir_data_percent = os.path.join(output_dir, f"data-sample-ratio-{data_percent:.4f}")
        utils.makedirs(output_dir_data_percent)

        index = generate_random_index(total_num=data.shape[0], choice_num=data.shape[0] * data_percent)
        label_true = label[index]

        np.save(os.path.join(output_dir_data_percent, 'data-index.npy'), index)
        np.save(os.path.join(output_dir_data_percent, 'clean-label.npy'), label_true)

        clusters = calculate_clusters(net, data[index], label_true, device, cluster_num=cluster_num_list[idx])

        # ------------------------------------------------------------------------------
        Output_str = f"Dataset Size:\t{len(index)}" + '\n'
        categories = np.unique(label_true)
        Output_str += f"Categories number: {len(categories)}" + '\n'
        transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
        for i in range(len(label_true)):
            transition_matrix[label_true[i]][label_true[i]] += 1
        info_file = open(os.path.join(output_dir_data_percent, 'info.txt'), 'w')
        info_file.write('-' * 10 + '\n')
        info_file.write(f"Original label matrix (noise rate={0:.4f}):\n")
        np.savetxt(info_file, X=transition_matrix, fmt="%i")
        info_file.write('-' * 10 + '\n')
        # ------------------------------------------------------------------------------

        for left_range in cluster_noise_rate_left_range:
            right_range = 2 * average_noise_rate - left_range
            assert right_range >= left_range

            print(
                f"Processing| Data percent: {data_percent:.4f}, cluster noise rate ({cluster_noise_type}): [{left_range:.4f}, {right_range:.4f}]")
            # cluster_noise_rate = np.random.uniform(left_range, right_range, size=cluster_num_list[idx])
            cluster_noise_rate = np.random.choice(np.array([left_range, right_range]), size=cluster_num_list[idx])

            label_noisy = np.empty_like(label_true)
            for i in range(len(cluster_noise_rate)):
                label_noisy[clusters == i] = generate_symmetric_noisy_label_cifar10(np.sum(clusters == i),
                                                                                    label_true[clusters == i],
                                                                                    cluster_noise_rate[i])

            np.save(os.path.join(output_dir_data_percent,
                                 f"noisy-rate-{average_noise_rate:.4f}-[{left_range:.4f}, {right_range:.4f}].npy"),
                    label_noisy)

            # ------------------------------------------------------------------------------
            info_file.write(
                f"Modified label matrix (noise rate={average_noise_rate:.4f}, [{left_range:.4f}, {right_range:.4f}]):\n")
            modified_label = label_noisy

            transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
            for i in range(len(label_true)):
                transition_matrix[label_true[i]][modified_label[i]] += 1
            np.savetxt(info_file, X=transition_matrix, fmt="%i")

            info_file.write('-' * 10 + '\n')
            # ------------------------------------------------------------------------------


def generate_class_dependent_label_noise(data_path="../cifar-10-batches-py",
                                         output_dir="../cifar-synthetic-processed/cifar-10/",
                                         data_percent_list=None, noise_rate_list=None):
    if data_percent_list is None:
        # data_percent_list = [0.002 * (i + 1) for i in range(4)] + \
        #                     [0.01 * (i + 1) for i in range(9)] + \
        #                     [0.1 * (i + 1) for i in range(10)]
        data_percent_list = [1.]

    average_noise_rate = 0.4
    if noise_rate_list is None:
        max_noise_rate = average_noise_rate
        noise_rate_list = [round(0.1 * i, 5) for i in range(int(max_noise_rate * 10) + 1)]

    class_non_uniform_set = np.zeros(10)
    class_non_uniform_set[np.random.permutation(np.arange(10))[:5]] = 1

    output_dir = os.path.join(output_dir, 'class_dependent_label')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    utils.makedirs(output_dir)

    data, label = data_divide.load_cifar10(data_path)
    label = np.array(label, dtype=int)

    for idx, data_percent in enumerate(data_percent_list):
        output_dir_data_percent = os.path.join(output_dir, f"data-sample-ratio-{data_percent:.4f}")
        utils.makedirs(output_dir_data_percent)

        index = generate_random_index(total_num=data.shape[0], choice_num=data.shape[0] * data_percent)
        label_true = label[index]

        np.save(os.path.join(output_dir_data_percent, 'data-index.npy'), index)
        np.save(os.path.join(output_dir_data_percent, 'clean-label.npy'), label_true)

        # ------------------------------------------------------------------------------
        Output_str = f"Dataset Size:\t{len(index)}" + '\n'
        categories = np.unique(label_true)
        Output_str += f"Categories number: {len(categories)}" + '\n'
        transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
        for i in range(len(label_true)):
            transition_matrix[label_true[i]][label_true[i]] += 1
        info_file = open(os.path.join(output_dir_data_percent, 'info.txt'), 'w')
        info_file.write('-' * 10 + '\n')
        info_file.write(f"Original label matrix (noise rate={0:.4f}):\n")
        np.savetxt(info_file, X=transition_matrix, fmt="%i")
        info_file.write('-' * 10 + '\n')
        # ------------------------------------------------------------------------------

        for noise_rate_left in noise_rate_list:
            noise_rate_right = 2 * average_noise_rate - noise_rate_left
            assert noise_rate_right >= noise_rate_left

            print(
                f"Processing| Data percent: {data_percent:.4f}, noise rate: [{noise_rate_left:.4f}, {noise_rate_right:.4f}]")

            label_noisy = np.empty_like(label_true)
            for i in range(10):
                index_class_i = label_true == i
                label_noisy[index_class_i] = generate_symmetric_noisy_label_cifar10(
                    np.sum(index_class_i),
                    label_true[index_class_i],
                    noise_rate_left if class_non_uniform_set[i] == 0 else noise_rate_right
                )

            np.save(os.path.join(output_dir_data_percent,
                                 f"noisy-rate-{average_noise_rate:.4f}-[{noise_rate_left:.4f}, {noise_rate_right:.4f}].npy"),
                    label_noisy)

            # ------------------------------------------------------------------------------
            info_file.write(
                f"Modified label matrix (noise rate={np.mean(label_true!=label_noisy):.4f}, [{noise_rate_left:.4f}, {noise_rate_right:.4f}]):\n")
            modified_label = label_noisy

            transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
            for i in range(len(label_true)):
                transition_matrix[label_true[i]][modified_label[i]] += 1
            np.savetxt(info_file, X=transition_matrix, fmt="%i")

            info_file.write('-' * 10 + '\n')
            # ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
def load_symmetric_data(data_dir='./data/cifar-10-batches-py',
                        index_root="../cifar-synthetic-processed/cifar-10/2022_11_12_15_38_7",
                        data_percent: str = 0., noise_rate: str = 1., verbose=False
                        ):
    noise_type = 'symmetric_label'
    index, clean_label, noisy_label = load_synthetic_index_and_label(index_root, noise_type, data_percent, noise_rate)

    data, _ = data_divide.load_cifar10(data_dir)
    data = np.array(data)[index]

    noise_rate = round(1 - np.mean(clean_label == noisy_label), 8)
    if verbose:
        print(f"Noise rate: {noise_rate}")

    return data, clean_label, noisy_label


# ------------------------------------------------------------------------------
# Funcs.
## Generate symmetric noisy label for cifar-10.
def generate_symmetric_noisy_label_cifar10(data_num, label, noise_rate):
    noise_label = []
    idx = np.arange(data_num)
    np.random.shuffle(idx)
    num_noise = int(noise_rate * data_num)
    noise_idx = idx[:num_noise]

    for i in range(data_num):
        if i in noise_idx:
            # nl = np.random.choice([j for j in range(10) if j != label[i]], replace=True)
            nl = np.random.randint(0, 10)
        else:
            nl = label[i]
        noise_label.append(nl)

    noise_label = np.array(noise_label, dtype=int)

    return noise_label


## Generate asymmetric noisy label for cifar-10.
def generate_asymmetric_noisy_label_cifar10(data_num, label, noise_rate):
    noise_label = []
    idx = np.arange(data_num)
    np.random.shuffle(idx)
    num_noise = int(noise_rate * data_num)
    noise_idx = idx[:num_noise]

    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise

    for i in range(data_num):
        if i in noise_idx:
            nl = transition[label[i]]
        else:
            nl = label[i]
        noise_label.append(nl)

    noise_label = np.array(noise_label, dtype=int)

    return noise_label


def load_synthetic_index_and_label(index_root, noise_type, data_percent, noise_rate):
    data_dir = os.path.join(index_root, noise_type, f"data-sample-ratio-{data_percent:.4f}")
    assert os.path.exists(data_dir), f"Data dir {data_dir} does not exist."

    index = np.load(os.path.join(data_dir, 'data-index.npy'))
    clean_label = np.load(os.path.join(data_dir, 'clean-label.npy'))
    noisy_label = np.load(os.path.join(data_dir, f"noisy-rate-{noise_rate:.4f}.npy"))

    return index, clean_label, noisy_label


def load_synthetic_cluster_dependent_label(index_root, noise_type, data_percent, noise_rate, low, high=None):
    high = 2 * noise_rate - low if high is None else high

    data_dir = os.path.join(index_root, noise_type, f"data-sample-ratio-{data_percent:.4f}")
    assert os.path.exists(data_dir), f"Data dir {data_dir} does not exist."

    index = np.load(os.path.join(data_dir, 'data-index.npy'))
    clean_label = np.load(os.path.join(data_dir, 'clean-label.npy'))
    noisy_label = np.load(os.path.join(data_dir, f"noisy-rate-{noise_rate:.4f}-[{low:.4f}, {high:.4f}].npy"))

    return index, clean_label, noisy_label


def generate_random_index(total_num: int, choice_num: int):
    index = np.arange(total_num)
    np.random.shuffle(index)
    index = index[:int(min(choice_num, total_num))]

    return index


def feature_extractor(net, data, label, device):
    dataset = SimpleDataset(data, label,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                            ]))
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    feature_arr = []
    target_arr = []
    with torch.no_grad():
        for iter_num, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            features = net(inputs, 'return_features')
            features = torch.flatten(features, 1).cpu()

            feature_arr.append(features)
            target_arr.append(targets)

    feature_arr = torch.cat(feature_arr)
    target_arr = torch.cat(target_arr)
    return feature_arr, target_arr


def calculate_clusters(net, data, label, device, cluster_num):
    features, _ = feature_extractor(net, data, label, device)
    print(cluster_num)
    clusters = cluster_kmeans(features, cluster_num=cluster_num)
    return clusters


if __name__ == "__main__":
    run_generate_synthetic_label_noise()
