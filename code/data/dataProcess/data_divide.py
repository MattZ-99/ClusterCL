# -*- coding: utf-8 -*-
# @Time : 2022/6/26 10:12
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
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import pickle
import shutil
import re


def seed_everything(seed: int = 0):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        res_dict = cPickle.load(fo, encoding='latin1')
    return res_dict


def save_pickle(obj: Any, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return True


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        content = pickle.load(f)

    return content


def load_cifar10(path):
    train_data = []
    train_label = []
    for n in range(1, 6):
        dpath = os.path.join(path, f"data_batch_{n}")
        data_dic = unpickle(dpath)
        train_data.append(data_dic['data'])
        train_label = train_label + data_dic['labels']

    train_data = np.concatenate(train_data)
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))

    return train_data, train_label


def load_cifar10_test(path):
    test_dic = unpickle(os.path.join(f"{path}", 'test_batch'))

    test_data = test_dic['data']
    test_data = test_data.reshape((10000, 3, 32, 32))
    test_data = test_data.transpose((0, 2, 3, 1))

    test_label = test_dic['labels']

    return test_data, test_label


def load_cifar_n_label(path, dataset='cifar10', noise_type='random_label1'):
    cifar_10_types = ['clean_label', 'aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3']
    cifar_100_types = ['clean_label', 'noisy_label', 'noisy_coarse_label', 'clean_coarse_label']

    if dataset == 'cifar10':
        assert noise_type in cifar_10_types
        path = os.path.join(path, 'CIFAR-10_human.pt')
    elif dataset == 'cifar100':
        assert noise_type in cifar_100_types
        path = os.path.join(path, 'CIFAR-100_human.pt')
    else:
        raise NotImplementedError

    noise_label = torch.load(path)
    assert isinstance(noise_label, dict)

    return np.squeeze(noise_label[noise_type]).tolist()


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
            modified_noise_label_dict["{:.4f}".format(noise_rate)] = modified_label

        self.noise_rate_adjustment_dict = modified_noise_label_dict
        return self.noise_rate_adjustment_dict

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

    def save_index_and_label(self, out_dir=None):
        np.save(os.path.join(out_dir, 'index.npy'), self.index)
        np.save(os.path.join(out_dir, 'noisy-rate-origin.npy'), self.get_current_label()[1])
        noisy_dict = self.noise_rate_adjustment_dict
        for key in noisy_dict:
            save_path = os.path.join(out_dir, f"noisy-rate-{key}.npy")
            np.save(save_path, noisy_dict[key])


def run_process_data():
    data_path = "../cifar-10-batches-py"
    cifar_n_path = "../cifar-n-labels/"

    data_percent_list = [0.002 * (i + 1) for i in range(4)] + [0.01 * (i + 1) for i in range(9)] + [0.1 * (i + 1) for i
                                                                                                    in range(10)]
    output_dir = "../cifar-n-processed/cifar-10/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    makedirs(output_dir)

    noise_type_list = ['aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3']
    save_pickle(noise_type_list, os.path.join(output_dir, 'noise-type-list.pickle'))

    for noise_type in noise_type_list:

        data, label = load_cifar10(data_path)
        noisy_label = load_cifar_n_label(cifar_n_path, noise_type=noise_type)

        data_process = CifarDataProcess(data, label, noisy_label)
        output_dir_noise_type = os.path.join(output_dir, f"{noise_type}_noise-rate-{data_process.noise_rate:.4f}")
        makedirs(output_dir_noise_type)
        save_pickle(data_percent_list, os.path.join(output_dir_noise_type, 'data-percent-list.pickle'))

        for data_percent in data_percent_list:
            output_dir_data_percent = output_dir_noise_type
            data_process.reset()
            data_process.data_num_sample(percent=data_percent)
            output_dir_data_percent = os.path.join(output_dir_data_percent,
                                                   f"data-sample-ratio-{data_percent:.4f}_noise-rate-{data_process.noise_rate:.4f}")
            makedirs(output_dir_data_percent)
            print(output_dir_data_percent)

            data_process.noise_label_sample()

            data_process.save_index_and_label(out_dir=output_dir_data_percent)


def get_data_type_list(root_dir: str) -> list:
    match_obj = None
    for sub_dir in os.listdir(root_dir):
        match_obj = re.match(r'.*\.pickle', sub_dir)
        if match_obj is not None:
            break
    if match_obj is not None:
        data_types = load_pickle(os.path.join(root_dir, match_obj.group(0)))
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
        data_percent_list = load_pickle(os.path.join(root_dir, match_obj.group(0)))
    else:
        raise FileNotFoundError

    data_percent_list = [round(i, 6) for i in data_percent_list]
    assert data_percent in data_percent_list

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
    index_name = 'index.npy'
    index_path = os.path.join(root_dir, index_name)
    assert os.path.exists(index_path)
    if verbose:
        print(f"Load index from {index_path}")
    index = np.load(index_path)

    noise_rate_name = "origin" if noise_rate == -1 else f"{noise_rate:.4f}"
    label_name = f"noisy-rate-{noise_rate_name}.npy"
    label_path = os.path.join(root_dir, label_name)
    assert os.path.exists(label_path)
    if verbose:
        print(f"Load noisy label from {label_path}")
    noisy_label = np.load(label_path)

    return index, noisy_label


def load_index_and_label(label_cifar_10_root, data_type, data_percent, noise_rate):
    data_type_list = get_data_type_list(label_cifar_10_root)
    if data_type not in data_type_list:
        raise FileNotFoundError

    noise_type_dir = get_data_type_dir(label_cifar_10_root, data_type)
    noisy_percent_dir = get_data_percent_dir(noise_type_dir, data_percent=data_percent)

    index, noisy_label = _load_index_and_label(noisy_percent_dir, noise_rate=noise_rate)

    return index, noisy_label


def load_data(data_dir: str, index_root: str, noise_type: str, data_percent: float, noise_rate: float, verbose=False):
    index, noisy_label = load_index_and_label(index_root, noise_type, data_percent, noise_rate)
    data, clean_label = load_cifar10(data_dir)
    data, clean_label = np.array(data)[index], np.array(clean_label)[index]

    noise_rate = round(1 - np.mean(clean_label == noisy_label), 8)
    if verbose:
        print(f"Noise rate: {noise_rate}")

    return data, clean_label, noisy_label


def run_load_data():
    label_cifar_10_root = "../cifar-n-processed/cifar-10"
    data_type = "worse_label"
    data_percent = 0.1
    noise_rate = 0.1

    index, noisy_label = load_index_and_label(label_cifar_10_root, data_type, data_percent, noise_rate)

    data_path = "../cifar-10-batches-py"
    data, clean_label = load_cifar10(data_path)
    data, clean_label = np.array(data)[index], np.array(clean_label)[index]

    noise_rate = round(1 - np.mean(clean_label == noisy_label), 8)
    print(noise_rate)


if __name__ == '__main__':
    run_process_data()
    # run_load_data()
