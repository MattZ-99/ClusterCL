# -*- coding: utf-8 -*-
# @Time : 2022/10/13 15:47
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
import argparse

import torch
import torch.nn as nn
from torch import optim

import tools.utils as utils
from datasets.DatasetGenerators import get_cifar10n_dataset_generator
from nets.wrapper_net import get_single_network, get_dual_networks, DualNets
from nets.loss_controller import LossController
from Modules import TrainModule

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ------------------------------------------------------------------------------
# Functions.

## Get args from command line.
def get_args(*para, **kwargs):
    parser = argparse.ArgumentParser(description='TRAIN CIFAR BASE.')

    # Basic parameters.
    group_basic = parser.add_argument_group("Basic parameters")
    group_basic.add_argument('--seed', default=0, type=int, help='Random seed.')
    group_basic.add_argument('--gpu', default=0, type=int, help='GPU ID.')

    # Dataset parameters.
    group_dataset = parser.add_argument_group("Dataset parameters")
    group_dataset.add_argument('--dataset', default='cifar10', type=str)
    group_dataset.add_argument('--data_path', default='./data/cifar-10-batches-py', type=str, help='path to dataset.')
    group_dataset.add_argument('--train_data_percent', default=1.0, type=float, help='Train data percent.')
    group_dataset.add_argument('--noise_mode', default='worse_label', type=str, help="noise mode.")
    group_dataset.add_argument('--noise_rate', default=-1, type=float, help='Noise rate.')

    group_dataset.add_argument('--num_classes', default=10, type=int, help='Number of classes.')

    group_dataset.add_argument('--num_batches', default=1000, type=int,
                               help='Set the maximum number of batches in one epoch.')
    group_dataset.add_argument('--num_samples', default=-1, type=int,
                               help='Set the maximum number of training samples across the whole training process.')
    group_dataset.add_argument('--cluster_dependent_noise_rate_low', default=0, type=float, help='Noise rate.')

    # Dataloader parameters.
    group_dataloader = parser.add_argument_group("Dataloader parameters")
    group_dataloader.add_argument('--batch_size', '-bs', default=64, type=int, help='train batch size.')
    group_dataloader.add_argument('--num_workers', default=8, type=int, help='number of data loader workers.')

    # Network parameters.
    group_network = parser.add_argument_group("Network parameters")
    group_network.add_argument('--net', default='pre_resnet34', type=str, help="Network name.")
    group_network.add_argument('--pretrained', action="store_true", default=False,
                               help="Network with pretrained weights.")
    group_network.add_argument('--net_nums', default='single_network', type=str, help="Number of networks.\n"
                                                                                      "single_network: single network.\n"
                                                                                      "dual_networks: dual networks.\n")

    # Training parameters.
    group_training = parser.add_argument_group("Training parameters")
    group_training.add_argument('--training_mode', default='0', type=str,
                                help="Training mode.\n"
                                     "-1: Develop mode.\n"
                                     "0: Train with clean data.\n"
                                     "0.1: Train with clean data, using mixup.\n"
                                     "0.2: Train with clean data, using mixup, consistency regularization.\n"
                                     "1: Train with noisy data.\n"
                                     "1.2: Train with noisy data, using mixup, consistency regularization.\n"
                                     "3.0: ClusterCL.\n"
                                     "3.0.1: ClusterCLPlusVersion, using mixup and fmix.\n"
                                     "3.1: ClusterCL Plus: using mixup, consistency regularization.\n"
                                     "3.2.1: ClusterCL UpdateVersion-1."
                                     "3.3: ClusterCL with supervised training."
                                     "3.4: ClusterCL with contrastive learning."
                                     "4: ProMix.\n"
                                     "5.0: DivideMix.\n"
                                     "5.1: DivideMix with supervised training.\n"
                                )

    group_training.add_argument('--epochs', default=300, type=int, help="Overall epochs to train.")
    group_training.add_argument('--epochs_warmup', default=10, type=int, help="Warmup epochs if necessary.")
    group_training.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate.')

    group_training.add_argument('--save_model', action="store_true", default=False, help="Whether to save the model.")
    group_training.add_argument('--save_interval', default=1000, type=int, help="Save model interval.")

    group_training.add_argument('--scheduler_mode', default='cosine', type=str,
                                help="Scheduler mode. Optional['cosine', 'step'].")

    # Remaining parameters.
    group_remaining = parser.add_argument_group("Remaining parameters")
    group_remaining.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    group_remaining.add_argument('--num_class', default=10, type=int)
    group_remaining.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    group_remaining.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    group_remaining.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    group_remaining.add_argument('--cluster_num', default=500, type=int)
    group_remaining.add_argument('--cluster_interval', default=30, type=int)
    group_remaining.add_argument('--cluster_method', default='kmeans', type=str)
    group_remaining.add_argument('--tau', default=0.99, type=float, help='high-confidence selection threshold')
    group_remaining.add_argument('--start_expand', default=100, type=int)

    group_remaining.add_argument('--n-views', default=2, type=int, metavar='N',
                                 help='Number of views for contrastive learning training.')

    group_remaining.add_argument('--dimension_reduction_method', default=None, type=str)
    group_remaining.add_argument('--dimension_reduction_components', default=100, type=int)

    _args = parser.parse_args(*para, **kwargs)

    return _args


## Make output dir path.
def make_output_dir(params):
    training_mode_dict = {
        '-1': 'develop_mode',
        '0': "TrainClean",
        '0.1': "TrainCleanMixup",
        '0.2': "TrainCleanMixupCR",
        '1': "TrainNoisy",
        '1.2': "TrainNoisyMixupCR",
        '3.0': "ClusterCL",
        '3.0.1': "ClusterCLPlusVersion",
        '3.1': "ClusterCLPlus",
        '3.2.1': "ClusterCL-UpdateVersion-1",
        '3.3': "ClusterCLWithSupervisedTraining",
        '3.4': "ClusterCLWithContrastiveLearning",
        '4': "ProMix",
        '5.0': "DivideMix",
        '5.1': "DivideMixWithSupervisedTraining",
    }

    res = os.path.join("./Outputs", utils.get_day_stamp(connector='/'), f"{params.dataset}",
                       f"{args.net_nums}-{training_mode_dict[params.training_mode]}",
                       f"class-num-{params.num_class}", params.noise_mode,
                       f"data-{params.train_data_percent:.2f}"
                       f"_noise-{params.noise_rate:.2f}"
                       f"_{params.net}_epochs-{params.epochs}"
                       f"_seed-{params.seed}_"
                       f"{utils.get_timestamp()}"
                       )
    return res


## Make dataset generator.
def make_dataset_generator(params):
    if params.dataset == "cifar10":
        if params.noise_mode in ["symmetric_label", "asymmetric_label"]:
            # index_root = "./data/cifar-synthetic-processed/cifar-10/2022_11_12_16_39_9"
            index_root = "./data/cifar-synthetic-processed/cifar-10/2022_11_12_22_4_59"
        elif params.noise_mode in ["cluster_dependent_label"]:
            # index_root = "./data/cifar-synthetic-processed/cifar-10/2022_11_13_15_5_38"
            # index_root = "./data/cifar-synthetic-processed/cifar-10/2022_11_14_21_42_21"
            index_root = "./data/cifar-synthetic-processed/cifar-10/2022_11_15_0_21_28"
        elif params.noise_mode in ["class_dependent_label"]:
            index_root = "./data/cifar-synthetic-processed/cifar-10/2022_11_15_23_30_50"
        else:
            index_root = "./data/cifar-n-processed/cifar-10/2022_9_3_10_51_27"

        res = get_cifar10n_dataset_generator(data_dir=params.data_path,
                                             index_root=index_root,
                                             noise_mode=params.noise_mode,
                                             noise_rate=params.noise_rate,
                                             data_percent=params.train_data_percent,
                                             cluster_dependent_noise_rate_low=args.cluster_dependent_noise_rate_low
                                             )
    else:
        raise NotImplementedError

    print(f"Train label noise rate: {res.train_label_noise_rate}")
    return res


## Create network.
def create_network(params):
    if params.net_nums == 'single_network':
        _net = get_single_network(params.net, params.num_class)
    elif params.net_nums == 'dual_networks':
        _net = get_dual_networks(params.net, params.num_class)
    else:
        raise NotImplementedError

    return _net


## Create optimizer.
def create_optimizer(params, _net) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
    if params.net_nums == 'single_network':
        assert isinstance(_net, nn.Module)
        _optimizer = optim.SGD(_net.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
        return _optimizer

    elif params.net_nums == 'dual_networks':
        assert isinstance(_net, DualNets)
        _optimizer_list = []
        for idx, _net_i in enumerate(_net):
            _optimizer = optim.SGD(_net_i.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
            _optimizer_list.append(_optimizer)
        return _optimizer_list

    else:
        raise NotImplementedError


## Create loss controller.
def create_loss_controller(params=None):
    return LossController()


# ------------------------------------------------------------------------------
# Main.
print("Running...")

## Boot.
print("Booting...")
args = get_args()

utils.print_args_wrapper(args)
utils.seed_everything(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}")

## Paths.
print("Making paths...")
output_dir = make_output_dir(args)
utils.makedirs(output_dir)
args.output_dir = output_dir
print(f"Output dir: {output_dir}")
utils.save_args(args, path=os.path.join(output_dir, 'args.json'), file_format='json')

## Initializations.
print("Initializing (data, network, optimizer, loss controller etc.)...")
dataset_generator = make_dataset_generator(args)
net = create_network(args).to(device)
optimizer = create_optimizer(args, net)
loss_controller = create_loss_controller()

# ------------------------------------------------------------------------------
# Training.

if args.net_nums == 'single_network':
    ## Training with clean data.
    if args.training_mode == '0':
        print("Initializing training module...")
        train_module = TrainModule.TrainCleanDataSingleNet(net=net, optimizer=optimizer,
                                                           loss_controller=loss_controller,
                                                           dataset_generator=dataset_generator,
                                                           device=device, args=args)

        print('\n' + '=' * 20 + 'Training with clean data' + '=' * 20)
        train_module.train(epochs=args.epochs)

    ## Training with clean data, using mixup.
    elif args.training_mode == '0.1':
        print("Initializing training module...")
        train_module = TrainModule.TrainCleanDataSingleNetWithMixup(net=net, optimizer=optimizer,
                                                                    loss_controller=loss_controller,
                                                                    dataset_generator=dataset_generator,
                                                                    device=device, args=args)
        print('\n' + '=' * 20 + 'Training with clean data (Mixup)' + '=' * 20)
        train_module.train(epochs=args.epochs)

    ## Training with clean data, using mixup and CR.
    elif args.training_mode == '0.2':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainCleanDataSingleNetWithMixupAndConsistencyRegularization
        train_module = TrainModuleClass(net=net,
                                        optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'Training with clean data (Mixup + CR)' + '=' * 20)
        train_module.train(epochs=args.epochs)

    ## Training with noisy data.
    elif args.training_mode == '1':
        # net.load_state_dict(torch.load("/home/zmt/MProject/NoiseLabel/draft34.pth")['state_dict'])
        # net.load_state_dict(torch.load("/home/zmt/MProject/NoiseLabel/imn34.pth")['state_dict'])

        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataSingleNet
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'Training with noisy data' + '=' * 20)
        train_module.train(epochs=args.epochs)

    ## Training with noisy data, using mixup and CR.
    elif args.training_mode == '1.2':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataSingleNetWithMixupAndCR
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'Training with noisy data (Mixup + CR)' + '=' * 20)
        train_module.train(epochs=args.epochs)

    else:
        raise NotImplementedError

elif args.net_nums == 'dual_networks':
    ## Training with clean data.
    if args.training_mode == '0':
        TrainModuleClass = TrainModule.TrainCleanDataDualNet
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)

        print('\n' + '=' * 20 + 'Training with clean data' + '=' * 20)
        train_module.train(epochs=args.epochs)

    ## Training with clean data, using mixup and CR.
    elif args.training_mode == '0.2':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainCleanDataDualNetWithMixupAndCR
        train_module = TrainModuleClass(net=net,
                                        optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'Training with clean data (Mixup + CR)' + '=' * 20)
        train_module.train(epochs=args.epochs)

    ## Training with noisy data, using mixup and CR.
    elif args.training_mode == '1.2':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetWithMixupAndCR
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'Training with noisy data (Mixup + CR)' + '=' * 20)
        train_module.train(epochs=args.epochs)

    elif args.training_mode == '3.0':
        print("Initializing training module...")
        net.net1.load_state_dict(torch.load("./draft18.pth")['state_dict'])

        TrainModuleClass = TrainModule.TrainNoisyDataDualNetClusterMix
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)

        print('\n' + '=' * 20 + 'ClusterCL' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '3.0.1':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetClusterMixPlusVersion2
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)

        print('\n' + '=' * 20 + 'ClusterCLPlusVersion, using mixup and fmix' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '3.1':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetClusterMixPlus
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'ClusterCL Plus (Mixup + CR)' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '3.2.1':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetClusterMixUpdateVersion1
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'ClusterCL UpdateVersion-1' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '3.3':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetClusterMixSupervised
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'ClusterCL Supervised' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '3.4':
        print("Initializing training module...")

        TrainModuleClass = TrainModule.TrainNoisyDataDualNetClusterMixWithContrastiveLoss
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)

        print('\n' + '=' * 20 + 'ClusterCLWithContrastiveLoss' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '4':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetProMix
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'ProMix (Mixup + CR)' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '5.0':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetDivideMix
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'DivideMix' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    elif args.training_mode == '5.1':
        print("Initializing training module...")
        TrainModuleClass = TrainModule.TrainNoisyDataDualNetDivideMixSupervised
        train_module = TrainModuleClass(net=net, optimizer=optimizer,
                                        loss_controller=loss_controller,
                                        dataset_generator=dataset_generator,
                                        device=device, args=args)
        print('\n' + '=' * 20 + 'DivideMix Supervised' + '=' * 20)
        train_module.train_with_warmup(epochs=args.epochs, epochs_warmup=args.epochs_warmup)

    else:
        raise NotImplementedError

else:
    raise NotImplementedError
