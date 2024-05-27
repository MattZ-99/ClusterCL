# -*- coding: utf-8 -*-
# @Time : 2022/6/24 18:53
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

import itertools
import math

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

from mixture import GaussianMixture

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])


def plot_gaussian(ax, mean, std, weight, **kwargs):
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    ax.plot(x, weight * stats.norm.pdf(x, mean, std), **kwargs)


def plot_data_distribution(data, model):
    fig, ax = plt.subplots()
    data = np.array(data)
    ax.hist(data, bins=200, alpha=0.7, density=True)

    means = np.squeeze(model.means_)
    covariances = np.squeeze(model.covariances_)
    weights = np.squeeze(model.weights_)

    ax.axvline(x=get_cross_point(model), ls='--', color='r')
    plot_gaussian(ax, means[0], math.sqrt(covariances[0]), weights[0])
    plot_gaussian(ax, means[1], math.sqrt(covariances[1]), weights[1])


def get_cross_point(model):
    val_min = np.min(gmm.means_)
    val_max = np.max(gmm.means_)
    intervals = np.linspace(val_min, val_max, 1000)
    prob = model.predict_proba(intervals.reshape(-1, 1))
    prob = prob[:, 0]

    for i in range(len(intervals)):
        if prob[i] > 0.5 and prob[i + 1] < 0.5:
            return intervals[i]


# Number of samples per component
n_samples = 5000
ratio = 0.7
# Generate random sample, two components
np.random.seed(0)
X_1 = np.random.randn(round(n_samples * ratio)) * 0.5 + 3
X_2 = np.random.randn(round(n_samples * (1-ratio))) * 2 + 5
X = np.concatenate([X_1, X_2])
X = X.reshape(-1, 1)

gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, fixed_weights=np.array([ratio, 1-ratio]))
gmm.fit(X)

print(f"means: {np.squeeze(gmm.means_)}")
print(f"covariances: {np.squeeze(gmm.covariances_)}")
print(f"weights: {np.squeeze(gmm.weights_)}")


# plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")
plot_data_distribution(X.reshape(-1), gmm)
plt.show()
