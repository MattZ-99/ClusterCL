# -*- coding: utf-8 -*-
# @Time : 2022/8/8 15:11
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

from evaluations.nmce_clusters import *
from evaluations.cluster_performance import *
from tools import utils
import os

# # use nmce to cluster cifar-10 data (unsupervised).
# run_nmce_cifar_10_clusters(
#     checkpoint_path='/home/mtzhang/projects/NoiseLabel/code/repositories/NMCE/exps/selfsup_resnet18cifar_cifar10_tcr_zw30/checkpoints/model-epoch800.pt',
#     n_clusters=20,
#     output_dir=os.path.join('Outputs/', 'evaluations/', utils.get_day_stamp(), 'nmce_clusters')
# )

# Check dividemix's dataset split in different clusters.
# Different clusters have different label noise rate.
# Hope: lower noise rate cluster, dividemix split better; while higher noise rate, performs worse.

# DivideMix w.o. cluster
# run_dividemix_performance_on_nmce_clusters(
#     clusters_info_path='./Outputs/evaluations/2022_8_8/nmce_clusters/nmce_clusters.pickle',
#     dividemix_pred_info_path='./repositories/DivideMix/Outputs/2022_8_6/class_nums_10/M_divideMix_no_noise_rate/not_know_label_noise/train_data_percent-1.0000/cifar10_worse_label_lu25.0_wp10_es300_noise-rate--1.0000/data-split-result/epoch_300.pickle'
# )

# DivideMix w. clusters
# run_dividemix_performance_on_nmce_clusters(
#     clusters_info_path='./Outputs/evaluations/2022_8_8/nmce_clusters/nmce_clusters.pickle',
#     dividemix_pred_info_path='./repositories/DivideMix/Outputs_220802/M_divideMix_divide_cluster/not_know_label_noise/train_data_percent-1.0000/cifar10_worse_label_lu25.0_wp10_es300_noise-rate--1.0000/data-split-result/epoch_269.pickle'
# )

# run_dividemix_clusters_performance_0_1(
#     output_dir=os.path.join('Outputs/', 'evaluations/', utils.get_day_stamp(), 'nmce_clusters')
# )

# run_dividemix_performance_on_nmce_clusters(
#     clusters_info_path='./Outputs/evaluations/2022_8_8/nmce_clusters/nmce_clusters_0_1.pickle',
#     dividemix_pred_info_path='./repositories/DivideMix/Outputs_220802/M_divideMix_no_noise_rate/not_know_label_noise/train_data_percent-0.1000/cifar10_worse_label_lu25.0_wp10_es300_noise-rate--1.0000/data-split-result/epoch_300.pickle'
# )
#
# run_dividemix_performance_on_nmce_clusters(
#     clusters_info_path='./Outputs/evaluations/2022_8_8/nmce_clusters/nmce_clusters_0_1.pickle',
#     dividemix_pred_info_path='./repositories/DivideMix/Outputs_220802/M_divideMix_divide_cluster/not_know_label_noise/train_data_percent-0.1000/cifar10_worse_label_lu25.0_wp10_es300_noise-rate--1.0000/data-split-result/epoch_233.pickle'
# )

# run_draft()


run_evaluation_cluster_performance_cifar()
