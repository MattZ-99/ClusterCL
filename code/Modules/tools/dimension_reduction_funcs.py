# -*- coding: utf-8 -*-
# @Time : 2022/11/4 13:46
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

from sklearn.decomposition import PCA, IncrementalPCA


def dimension_reduction_pca(feature_arr, dimension_reduction_num=2):
    pca = PCA(n_components=dimension_reduction_num)
    return pca.fit_transform(feature_arr)


def dimension_reduction_ipca(feature_arr, dimension_reduction_num=2):
    pca = IncrementalPCA(n_components=dimension_reduction_num)
    return pca.fit_transform(feature_arr)


def dimension_reduction_lda(feature_arr, dimension_reduction_num):
    raise NotImplementedError


def dimension_reduction_tsne(feature_arr, dimension_reduction_num):
    raise NotImplementedError


def dimension_reduction_umap(feature_arr, dimension_reduction_num):
    raise NotImplementedError
