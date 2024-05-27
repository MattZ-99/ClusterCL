# -*- coding: utf-8 -*-
# @Time : 2022/10/15 11:18
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

import numpy as np


def get_average_value(vals, last_n=1):
    if len(vals) < last_n:
        return np.mean(vals)
    return np.mean(vals[-last_n:])


def linear_ramp_up(current, ramp_up_length):
    """Linear ramp_up"""
    assert current >= 0 and ramp_up_length >= 0
    if current >= ramp_up_length:
        return 1.0
    else:
        return current / ramp_up_length


def calculate_precision_recall_f1(y_true, y_pred):
    """Calculate precision, recall, f1 score.

    Args:
        y_true (np.ndarray): True label.
        y_pred (np.ndarray): Predicted label.

    Returns:
        tuple: (precision, recall, f1)
    """
    assert y_true.shape == y_pred.shape
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
