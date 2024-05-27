import os
import pickle
import re
import shutil
import time
from typing import Any

import imageio
import matplotlib.pyplot as plt
import numpy as np


def rm_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_args(args, path, file_format=None):
    save_dict = vars(args)

    if file_format is None:
        save_a_dict(save_dict, path)
    elif file_format == 'json':
        save_json(save_dict, path)
    elif file_format == 'yaml':
        save_yaml(save_dict, path)
    else:
        raise ValueError(f"file_format {file_format} is not supported.")


def save_a_dict(save_dict, path):
    with open(path, 'w') as f:
        for key in save_dict:
            f.write(f"{key}: {save_dict[key]}\n")


def save_json(data, path):
    import json
    with open(path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


def save_yaml(data, path):
    import yaml
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=True, indent=4)


def get_timestamp(file_name=None):
    localtime = time.localtime(time.time())
    date_time = "{}_{}_{}_{}_{}_{}".format(localtime.tm_year, localtime.tm_mon, localtime.tm_mday, localtime.tm_hour,
                                           localtime.tm_min, localtime.tm_sec)
    if file_name:
        return file_name + '_' + date_time
    return date_time


def get_day_stamp(file_name=None, connector='_'):
    localtime = time.localtime(time.time())
    date_time = f"{localtime.tm_year}{connector}{localtime.tm_mon}{connector}{localtime.tm_mday}"
    if file_name:
        return file_name + connector + date_time
    return date_time


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


# noinspection SpellCheckingInspection
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class ValueStat:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, v=0, n=1):
        self.value += v * n
        self.count += n

    def reset(self):
        self.value = 0
        self.count = 0

    def get_sum(self):
        return self.value

    def get_avg(self):
        if self.count == 0:
            return -1
        return self.value / self.count


class ValueMinMaxStat(object):
    def __init__(self, min_val=0, max_val=0):
        self.min_val = min_val
        self.max_val = max_val
        self.update_max_flag = 0
        self.update_min_flag = 0

    def set_min_val(self, val):
        self.min_val = val

    def set_max_val(self, val):
        self.max_val = val

    def update(self, val):
        if self.min_val > val:
            self.min_val = val
            self.update_min_flag = 1
        else:
            self.update_min_flag = 0
        if self.max_val < val:
            self.max_val = val
            self.update_max_flag = 1
        else:
            self.update_max_flag = 0

    def get_max_update_flag(self):
        return self.update_max_flag

    def get_min_update_flag(self):
        return self.update_min_flag


class ValueListStat:
    def __init__(self, shape=(1,)):
        self.shape = shape
        self.value_arr = np.zeros(shape)
        self.count = 0.

    def update(self, vl: list | np.ndarray | None = None, n=1):
        if vl is None:
            vl = np.zeros(self.shape)

        vl = np.array(vl)
        self.value_arr += vl * n

        self.count += n

    def reset(self):
        self.value_arr = np.zeros(self.shape)
        self.count = 0.

    def get_sum(self):
        return self.value_arr.tolist()

    def get_avg(self):
        if self.count == 0:
            return np.zeros(self.shape) - 1
        return (self.value_arr / self.count).tolist()


class ValuesVisual:
    def __init__(self):
        self.values = []

    def add_value(self, val):
        self.values.append(val)

    def __len__(self):
        return len(self.values)

    def plot(self, output_path, title="Title", xlabel="X-axis", ylabel="Y-axis"):
        length = len(self)
        if length == 0:
            return -1
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x_axis = [i for i in range(length)]
        ax.plot(x_axis, self.values, color='tab:blue')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(output_path)
        plt.close()


def create_gif(image_list, gif_name, duration=0.1, verbose=0):
    frames = []
    for image_name in image_list:
        if verbose == 1:
            print(image_name)
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def string_list_match(data_list, match_str, match_or_search=0):
    new_data_list = list()
    if isinstance(match_str, str):
        match_str = [match_str]
    elif isinstance(match_str, list) and len(match_str) > 0 and isinstance(match_str[0], str):
        pass
    else:
        raise TypeError("Input is not valid.")

    if isinstance(data_list, str):
        data_list = [data_list]

    if match_or_search:
        func_match = re.search
    else:
        func_match = re.match
    for data in data_list:
        for ms in match_str:
            if func_match(ms, data) is not None:
                new_data_list.append(data)
                break
    return new_data_list


def save_pickle(obj: Any, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return True


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        content = pickle.load(f)

    return content


def demo():
    print('demo')


def print_args(args):
    save_dict = vars(args)
    for key in save_dict:
        print(f"{key}: {save_dict[key]}")


def print_args_sorted(args):
    args_dict = vars(args)
    key_list_sorted = sorted(args_dict.keys())
    for key in key_list_sorted:
        print(f"{key}: {args_dict[key]}")


def print_args_wrapper(args):
    print("\n" + "=" * 20 + " Arguments " + "=" * 20)
    print_args_sorted(args)
    print("=" * 51 + "\n")
