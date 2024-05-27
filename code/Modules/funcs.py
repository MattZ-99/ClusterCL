# -*- coding: utf-8 -*-
# @Time : 2022/8/24 13:49
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
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN


def cluster_kmeans(data, cluster_num=100):
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    clusters = kmeans.fit_predict(data)

    return clusters


def cluster_kmeans_plusplus(data, cluster_num=100):
    kmeans = KMeans(n_clusters=cluster_num, init='k-means++', random_state=0)
    clusters = kmeans.fit_predict(data)

    return clusters


def cluster_SpectralClustering(data, cluster_num=100):
    spectral = SpectralClustering(n_clusters=cluster_num, random_state=0, assign_labels='cluster_qr')
    clusters = spectral.fit_predict(data)

    return clusters


# Ward hierarchical clustering
def cluster_AgglomerativeClustering(data, cluster_num=100):
    agg = AgglomerativeClustering(n_clusters=cluster_num)
    clusters = agg.fit_predict(data)

    return clusters


def cluster_dbscan(data, cluster_num=100):
    dbscan = DBSCAN(eps=0.3, min_samples=len(data) // (2 * cluster_num))
    clusters = dbscan.fit_predict(data)
    clusters += np.min(clusters)

    return clusters


def get_average_value(vals, last_n=1):
    if len(vals) < last_n:
        return np.mean(vals)
    return np.mean(vals[-last_n:])
