# -*- coding: utf-8 -*-
# @Time : 2022/7/3 19:31
# @Author : Mengtian Zhang
# @E-mail : zhangmengtian@sjtu.edu.cn
# @Version : v-dev-0.0
# @License : MIT License
# @Copyright : Copyright 2022, Mengtian Zhang
# @Function 

"""Select part categories from the overall dataset.

Description.----------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

Example:

"""

import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../tools"))
import utils

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    from data_divide import load_cifar10, load_cifar_n_label
    from process_data import load_synthetic_index_and_label
else:
    from .data_divide import load_cifar10, load_cifar_n_label
    from .process_data import load_synthetic_index_and_label, load_synthetic_cluster_dependent_label

cifar_10_categories_name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
cifar_10n_noise_type_list = ['aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3']
cifar_10_synthetic_noise_type_list = ['symmetric_label', 'asymmetric_label', 'cluster_dependent_label', 'class_dependent_label']


class CifarDataProcess:
    def __init__(self, data, clean_label, noisy_label):
        super(CifarDataProcess, self).__init__()
        self.noisy_label = np.array(noisy_label)
        self.clean_label = np.array(clean_label)
        self.data = data

        self.index = np.array([i for i in range(len(self.data))])
        self.data_sample_rate = 1.0
        self.noise_rate_adjustment_dict = None

    def reset(self):
        self.index = np.array([i for i in range(len(self.data))])
        self.data_sample_rate = 1.0
        self.noise_rate_adjustment_dict = None

    def __len__(self):
        return len(self.index)

    @property
    def noise_rate(self):
        _clean_label, _noisy_label = self.get_current_label()

        clean_num = np.sum(_noisy_label == _clean_label)
        total_num = len(self)

        return (total_num - clean_num) / total_num

    def get_current_label(self):
        _noisy_label = self.noisy_label[self.index]
        _clean_label = self.clean_label[self.index]

        return _clean_label, _noisy_label

    def get_current_data(self):
        return self.data[self.index]

    def noise_label_sample(self, noise_rate_list=None):
        _clean_label, _noisy_label = self.get_current_label()
        assert isinstance(_clean_label, np.ndarray) and isinstance(_noisy_label, np.ndarray)

        noise_pos = (1 - (_clean_label == _noisy_label)).nonzero()[0]
        noise_pos = np.random.permutation(noise_pos)
        max_noise_rate = self.noise_rate

        if noise_rate_list is None:
            noise_rate_list = [round(0.01 * i, 5) for i in range(int(max_noise_rate * 100) + 1)]

        modified_noise_label_dict = {}
        for noise_rate in noise_rate_list:
            assert noise_rate <= max_noise_rate
            c_noise_pos = noise_pos[:int(len(_clean_label) * noise_rate)]

            flag_noise = np.zeros(len(_clean_label), dtype=bool)
            flag_noise[c_noise_pos] = True

            modified_label = np.where(flag_noise, _noisy_label, _clean_label)
            modified_noise_label_dict[noise_rate] = modified_label

        self.noise_rate_adjustment_dict = modified_noise_label_dict
        return self.noise_rate_adjustment_dict

    def output_data_info_statistic(self, output_path=None):
        Output_str = f"Dataset Size:\t{len(self)}" + '\n'

        clean_label, noisy_label = self.get_current_label()
        categories = np.unique(clean_label)
        Output_str += f"Categories number: {len(categories)}" + '\n'

        transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
        for i in range(len(clean_label)):
            transition_matrix[clean_label[i]][noisy_label[i]] += 1

        with open(output_path, 'w') as f:
            f.write(Output_str)
            f.write('-' * 10 + '\n')
            f.write(f"Original label matrix (noise rate={self.noise_rate:.4f}):\n")
            np.savetxt(f, X=transition_matrix, fmt="%i")
            f.write('-' * 10 + '\n')

            for key in self.noise_rate_adjustment_dict:
                f.write(f"Modified label matrix (noise rate={key:.4f}):\n")
                modified_label = self.noise_rate_adjustment_dict[key]

                transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
                for i in range(len(clean_label)):
                    transition_matrix[clean_label[i]][modified_label[i]] += 1
                np.savetxt(f, X=transition_matrix, fmt="%i")

                f.write('-' * 10 + '\n')

    def data_num_sample(self, percent: float):
        assert (percent > 0) and (percent <= 1.0)

        self.data_sample_rate *= percent

        if percent == 1.0:
            return

        _clean_label, _noisy_label = self.get_current_label()
        train_index, _ = train_test_split(self.index,
                                          train_size=percent, random_state=1234,
                                          stratify=_clean_label
                                          )

        self.index = train_index

    def save_index_and_label(self, out_dir=None, abs_index=None):
        np.save(os.path.join(out_dir, 'data-index.npy'), abs_index[self.index])
        np.save(os.path.join(out_dir, 'clean-label.npy'), self.get_current_label()[0])
        np.save(os.path.join(out_dir, 'noisy-rate-origin.npy'), self.get_current_label()[1])
        noisy_dict = self.noise_rate_adjustment_dict
        for key in noisy_dict:
            save_path = os.path.join(out_dir, f"noisy-rate-{key:.4f}.npy")
            np.save(save_path, noisy_dict[key])


def categories_selection(label, selected_categories=None) -> np.ndarray:
    if selected_categories is None:
        selected_categories = [3, 5]

    label = np.array(label)
    index = np.zeros_like(label, dtype=bool)

    for c in selected_categories:
        index |= (label == c)

    return index


def process_noisy_label(clean_label, noisy_label, selected_categories=None):
    if selected_categories is None:
        selected_categories = [3, 5]

    noisy_label_new = []
    for i in range(len(noisy_label)):
        if noisy_label[i] in selected_categories:
            noisy_label_new.append(noisy_label[i])
        else:
            noisy_label_new.append(clean_label[i])

    noisy_label = np.array(noisy_label_new)

    noisy_label_new = []
    clean_label_new = []
    for i in range(len(noisy_label)):
        clean_label_new.append(selected_categories.index(clean_label[i]))
        noisy_label_new.append(selected_categories.index(noisy_label[i]))

    return np.array(clean_label_new), np.array(noisy_label_new)


def run_data_class_selection():
    data_path = "../cifar-10-batches-py"
    cifar_n_path = "../cifar-n-labels/"

    output_dir = f"../cifar-n-processed/cifar-10-selected-two-categories-3-5/{utils.get_timestamp()}"
    selected_categories = [3, 5]
    # output_dir = f"../cifar-n-processed/cifar-10/{utils.get_timestamp()}"
    # selected_categories = [i for i in range(10)]
    utils.makedirs(output_dir)

    data_percent_list = [0.002 * (i + 1) for i in range(4)] + [0.01 * (i + 1) for i in range(9)] + [0.1 * (i + 1) for i
                                                                                                    in range(10)]
    data_percent_list = [round(i, 6) for i in data_percent_list]

    noise_type_list = ['aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3']
    utils.save_pickle(noise_type_list, os.path.join(output_dir, 'noise-type-list.pickle'))

    for noise_type in noise_type_list:

        data, label = load_cifar10(data_path)
        label = np.array(label)
        noisy_label = load_cifar_n_label(cifar_n_path, noise_type=noise_type)
        noisy_label = np.array(noisy_label)

        index = categories_selection(label, selected_categories=selected_categories)
        data, label, noisy_label = data[index], label[index], noisy_label[index]

        label, noisy_label = process_noisy_label(label, noisy_label, selected_categories=selected_categories)

        data_process = CifarDataProcess(data, label, noisy_label)

        output_dir_noise_type = os.path.join(output_dir, f"{noise_type}_noise-rate-{data_process.noise_rate:.4f}")
        utils.makedirs(output_dir_noise_type)
        utils.save_pickle(data_percent_list, os.path.join(output_dir_noise_type, 'data-percent-list.pickle'))

        for data_percent in data_percent_list:
            data_process.reset()
            data_process.data_num_sample(percent=data_percent)
            data_process.noise_label_sample()

            output_dir_data_percent = os.path.join(output_dir_noise_type,
                                                   f"data-sample-ratio-{data_percent:.4f}_noise-rate-{data_process.noise_rate:.4f}")
            utils.makedirs(output_dir_data_percent)

            data_process.save_index_and_label(out_dir=output_dir_data_percent, abs_index=index.nonzero()[0])
            data_process.output_data_info_statistic(output_path=os.path.join(output_dir_data_percent, 'info.txt'))


def get_data_type_list(root_dir: str) -> list:
    match_obj = None
    for sub_dir in os.listdir(root_dir):
        match_obj = re.match(r'.*\.pickle', sub_dir)
        if match_obj is not None:
            break
    if match_obj is not None:
        data_types = utils.load_pickle(os.path.join(root_dir, match_obj.group(0)))
    else:
        raise FileNotFoundError

    return data_types


def get_data_type_dir(root_dir: str, data_type: str) -> str:
    match_obj = None
    for sub_dir in os.listdir(root_dir):
        match_obj = re.match(rf'{data_type}.*', sub_dir)
        if match_obj is not None:
            break

    if match_obj is None:
        raise FileNotFoundError

    return os.path.join(root_dir, match_obj.group())


def get_data_percent_dir(root_dir: str, data_percent: float):
    match_obj = None
    for sub_dir in os.listdir(root_dir):
        match_obj = re.match(r'.*\.pickle', sub_dir)
        if match_obj is not None:
            break
    if match_obj is not None:
        data_percent_list = utils.load_pickle(os.path.join(root_dir, match_obj.group(0)))
    else:
        raise FileNotFoundError

    data_percent_list = [round(i, 6) for i in data_percent_list]
    assert data_percent in data_percent_list, f'{data_percent} not in {data_percent_list}'

    match_obj = None
    for sub_dir in os.listdir(root_dir):
        match_obj = re.match(rf"data-sample-ratio-{data_percent:.4f}.*", sub_dir)
        if match_obj is not None:
            break

    if match_obj is not None:
        sub_name = match_obj.group()
    else:
        raise FileNotFoundError

    return os.path.join(root_dir, sub_name)


def _load_index_and_label(root_dir: str, noise_rate: float, verbose=False):
    index_name = 'data-index.npy'
    index_path = os.path.join(root_dir, index_name)
    assert os.path.exists(index_path)
    if verbose:
        print(f"Load index from {index_path}")
    index = np.load(index_path)

    clean_path = os.path.join(root_dir, "clean-label.npy")
    assert os.path.exists(clean_path)
    if verbose:
        print(f"Load clean label from {clean_path}")
    clean_label = np.load(clean_path)

    noise_rate_name = "origin" if noise_rate == -1 else f"{noise_rate:.4f}"
    label_name = f"noisy-rate-{noise_rate_name}.npy"
    label_path = os.path.join(root_dir, label_name)
    assert os.path.exists(label_path)
    if verbose:
        print(f"Load noisy label from {label_path}")
    noisy_label = np.load(label_path)

    return index, clean_label, noisy_label


def load_index_and_label(label_cifar_10_root, data_type, data_percent, noise_rate):
    data_type_list = get_data_type_list(label_cifar_10_root)
    if data_type not in data_type_list:
        raise FileNotFoundError

    noise_type_dir = get_data_type_dir(label_cifar_10_root, data_type)
    noisy_percent_dir = get_data_percent_dir(noise_type_dir, data_percent=data_percent)

    index, clean_label, noisy_label = _load_index_and_label(noisy_percent_dir, noise_rate=noise_rate)

    return index, clean_label, noisy_label


def run_load_data():
    label_cifar_10_root = "../cifar-n-processed/cifar-10-slected-two-categories-3-5/2022_7_4_11_51_3"
    data_type = "worse_label"
    data_percent = 0.01
    noise_rate = 0.1

    index, clean_label, noisy_label = load_index_and_label(label_cifar_10_root, data_type, data_percent, noise_rate)

    data_path = "../../../data/cifar-10-batches-py"
    data, _ = load_cifar10(data_path)
    data = np.array(data)[index]

    noise_rate = round(1 - np.mean(clean_label == noisy_label), 8)
    print(noise_rate)


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        res_dict = cPickle.load(fo, encoding='latin1')
    return res_dict


def load_cifar10_test(path, selected_categories=None):
    if selected_categories is None:
        selected_categories = [3, 5]

    test_dic = unpickle(os.path.join(f"{path}", 'test_batch'))
    test_data = test_dic['data']
    test_data = test_data.reshape((10000, 3, 32, 32))
    test_data = test_data.transpose((0, 2, 3, 1))
    test_label = test_dic['labels']
    test_label = np.array(test_label)

    index = np.zeros_like(test_label, dtype=bool)

    for c in selected_categories:
        index |= (test_label == c)
    test_label = test_label[index]
    test_data = test_data[index]

    test_label_new = []
    for i in range(len(test_label)):
        test_label_new.append(selected_categories.index(test_label[i]))

    return test_data, np.array(test_label_new)


def load_data_test(path):
    test_dic = unpickle(os.path.join(f"{path}", 'test_batch'))
    test_data = test_dic['data']
    test_data = test_data.reshape((10000, 3, 32, 32))
    test_data = test_data.transpose((0, 2, 3, 1))
    test_label = test_dic['labels']
    test_label = np.array(test_label)

    return test_data, test_label


def load_data(data_dir: str, index_root: str, noise_type: str, data_percent: float, noise_rate: float, verbose=False, *args, **kwargs):
    if noise_type in cifar_10n_noise_type_list:
        index, clean_label, noisy_label = load_index_and_label(index_root, noise_type, data_percent, noise_rate)
    elif noise_type in ["cluster_dependent_label", "class_dependent_label"]:
        index, clean_label, noisy_label = load_synthetic_cluster_dependent_label(index_root, noise_type, data_percent, noise_rate, low=kwargs.get("cluster_dependent_noise_rate_low", 0))
    elif noise_type in cifar_10_synthetic_noise_type_list:
        index, clean_label, noisy_label = load_synthetic_index_and_label(index_root, noise_type, data_percent, noise_rate)
    else:
        raise ValueError(f"noise_type {noise_type} not in {cifar_10n_noise_type_list + cifar_10_synthetic_noise_type_list}")

    data, _ = load_cifar10(data_dir)
    data = np.array(data)[index]

    noise_rate = round(1 - np.mean(clean_label == noisy_label), 8)
    if verbose:
        print(f"Noise rate: {noise_rate}")

    return data, clean_label, noisy_label


def load_synthetic_data(data_dir: str, index_root: str, data_percent: float, noise_rate: float, verbose=False):
    index_dir = os.path.join(index_root, f"synthetic_noise/data-sample-ratio-{data_percent:.4f}")
    assert os.path.exists(index_dir), f"Data dir ({index_dir}) does not exist."

    index = np.load(os.path.join(index_dir, "data-index.npy"))
    clean_label = np.load(os.path.join(index_dir, "clean-label.npy"))
    noisy_label = np.load(os.path.join(index_dir, f"noisy-rate-{noise_rate:.4f}.npy"))

    data, _ = load_cifar10(data_dir)
    data = np.array(data)[index]

    noise_rate = round(1 - np.mean(clean_label == noisy_label), 8)
    if verbose:
        print(f"Noise rate: {noise_rate}")

    return data, clean_label, noisy_label


def generate_synthetic_noise(data_dir, index_root, noise_type, data_percent, class_num=2,
                             noise_rate_arr=None):
    if noise_rate_arr is None:
        noise_rate_arr = np.arange(0., 0.41, 0.01)

    train_data, train_label, _ = load_data(
        data_dir=data_dir,
        index_root=index_root,
        noise_type=noise_type,
        data_percent=data_percent,
        noise_rate=-1
    )

    data_num = train_data.shape[0]
    train_data = np.reshape(train_data, (data_num, -1))
    data_dim = train_data.shape[1]

    noisy_label_dict = {}

    for noise_rate in noise_rate_arr:
        # Sample instance flip rates
        flip_rates = np.clip(noise_rate + 0.1 * np.random.randn(data_num), 0, 1)
        omega = np.random.randn(class_num, data_dim, class_num)

        label_noisy_arr = []

        for idx in range(data_num):
            data = train_data[idx]
            label = train_label[idx]

            prob = np.matmul(data, omega[label])
            prob[label] = -np.inf
            prob = torch.from_numpy(prob)
            prob = F.softmax(prob, dim=0).numpy()

            prob = flip_rates[idx] * prob
            prob[label] = 1 - flip_rates[idx]

            label_noisy = np.random.choice(class_num, p=prob)
            label_noisy_arr.append(label_noisy)

        label_noisy_arr = np.array(label_noisy_arr)

        noise_rate_real = np.mean(np.not_equal(train_label, label_noisy_arr))
        noisy_label_dict[noise_rate] = label_noisy_arr

        # print(f"Noise rate: {noise_rate}, real noise rate: {noise_rate_real}")

    return train_data, train_label, noisy_label_dict


def run_generate_synthetic_noise():
    index_root = '../cifar-n-processed/cifar-10-selected-two-categories-3-5/2022_9_5_23_31_6'

    data_percent_list = [0.002 * (i + 1) for i in range(4)] + [0.01 * (i + 1) for i in range(9)] + [0.1 * (i + 1) for i
                                                                                                    in range(10)]
    data_percent_list = [round(i, 6) for i in data_percent_list]

    save_dir = os.path.join(index_root, 'synthetic_noise')
    utils.rm_dir(save_dir)

    for percent in data_percent_list:
        print(f"Data percent: {percent}")

        save_subdir = os.path.join(save_dir, f"data-sample-ratio-{percent:.4f}")
        utils.makedirs(save_subdir)

        train_data, train_label, noisy_label_dict = generate_synthetic_noise(
            data_dir='../cifar-10-batches-py',
            index_root=index_root,
            noise_type="worse_label",
            data_percent=percent
        )

        index, clean_label, _ = load_index_and_label(index_root, "worse_label", percent, -1)
        assert np.all(train_label == clean_label)

        np.save(os.path.join(save_subdir, 'data-index.npy'), index)
        np.save(os.path.join(save_subdir, 'clean-label.npy'), clean_label)
        for noise_rate in noisy_label_dict.keys():
            np.save(os.path.join(save_subdir, f'noisy-rate-{noise_rate:.4f}.npy'), noisy_label_dict[noise_rate])


if __name__ == '__main__':
    # run_load_data()

    # Process data
    # run_data_class_selection()

    # Generate synthetic noise
    run_generate_synthetic_noise()

