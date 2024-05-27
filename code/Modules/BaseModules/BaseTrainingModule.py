# -*- coding: utf-8 -*-
# @Time : 2022/10/14 13:43
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
import sys

import torch
from torch import nn

from ..tools import metrics, funcs

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from tools import utils


class BaseTrainingModule:
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device=None, args=None):
        self.dataset_generator = dataset_generator
        self.net = net
        self.optimizer = optimizer
        self.loss_controller = loss_controller
        self.args = args

        if device is None:
            device = torch.device("cpu")
        self.device = device

        self.history_info = {}

        # Output paths.
        self.log_train_path = os.path.join(args.output_dir, 'log_train.log')
        self.log_test_path = os.path.join(args.output_dir, 'log_test.log')
        self.logger_train = open(self.log_train_path, 'w')
        self.logger_test = open(self.log_test_path, 'w')

        if self.args.save_model:
            self.model_save_dir = os.path.join(args.output_dir, 'models')
            utils.makedirs(self.model_save_dir)

    def train_basic_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, epoch, *args, **kwargs):
        stat_dict = {
            'epoch': epoch,
            'net': self.net.state_dict(),
        }

        save_path = os.path.join(self.model_save_dir, f"epoch_{epoch}.pth")

        torch.save(stat_dict, save_path)

    def load_model(self, load_path='./model.pth', *args, **kwargs):
        stat_dict = torch.load(load_path)
        self.net.load_state_dict(stat_dict['net'])
        return stat_dict['epoch']


class TrainSingleNetModule(BaseTrainingModule):

    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        assert isinstance(net, nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)

        super(TrainSingleNetModule, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_basic_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def test_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        # Parameters.
        args = self.args
        device = self.device
        net = self.net
        net.eval()
        loss_fn = self.loss_controller.get_cross_entropy_loss()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)

                # Forward.
                outputs = net(data)
                loss_ce = loss_fn(outputs, targets)

                # Log.
                loss_val.update(loss_ce.item())
                acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))

                acc_val_top_1.update(acc_1.item())
                acc_val_top_5.update(acc_5.item())

        # Result dict.
        result_dict = {
            'epoch': epoch,
            'loss_ce': loss_val.get_avg(),
            'acc_top_1': acc_val_top_1.get_avg(),
            'acc_top_5': acc_val_top_5.get_avg()
        }
        if kwargs.get('test_data', False):
            self.history_info.setdefault('test_acc', []).append(result_dict['acc_top_1'])
            result_dict['avg_acc'] = funcs.get_average_value(self.history_info['test_acc'], 10)

        return result_dict


class TrainDualNetModule(BaseTrainingModule):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        assert isinstance(net, nn.Module)
        assert isinstance(optimizer, list) and len(optimizer) == 2 and \
               isinstance(optimizer[0], torch.optim.Optimizer) and isinstance(optimizer[1], torch.optim.Optimizer)

        super(TrainDualNetModule, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def train_basic_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        """Test one epoch.

        :param epoch:
        :param data_loader:
        :param args:
        :param kwargs:
        :return: result_dict = { 'epoch': ...,
                    'net1': {'loss_ce': ..., 'acc_top_1': ..., 'acc_top_5': ...},
                    'net2': {...},
                    'net_mean': {'acc_top_1':..., 'acc_top_5':...}
                }
        """

        # Parameters.
        args = self.args
        device = self.device

        net1 = self.net.net1
        net2 = self.net.net2
        net1.eval()
        net2.eval()

        loss_fn = self.loss_controller.get_cross_entropy_loss()

        loss_val_net1, loss_val_net2 = utils.ValueStat(), utils.ValueStat()
        acc_val_top_1_net1, acc_val_top_1_net2, acc_val_top_1_net_mean = utils.ValueStat(), utils.ValueStat(), utils.ValueStat()
        acc_val_top_5_net1, acc_val_top_5_net2, acc_val_top_5_net_mean = utils.ValueStat(), utils.ValueStat(), utils.ValueStat()

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)

                # Forward.
                outputs1 = net1(data)
                outputs2 = net2(data)
                outputs_mean = (outputs1 + outputs2) / 2  # Average of two networks.

                loss_ce1 = loss_fn(outputs1, targets)
                loss_ce2 = loss_fn(outputs2, targets)

                # Log.
                loss_val_net1.update(loss_ce1.item())
                loss_val_net2.update(loss_ce2.item())

                ## Acc for net1.
                acc_1_net1, acc_5_net1 = metrics.accuracy(outputs1, targets, topk=(1, 5))
                acc_val_top_1_net1.update(acc_1_net1.item())
                acc_val_top_5_net1.update(acc_5_net1.item())

                ## Acc for net2.
                acc_1_net2, acc_5_net2 = metrics.accuracy(outputs2, targets, topk=(1, 5))
                acc_val_top_1_net2.update(acc_1_net2.item())
                acc_val_top_5_net2.update(acc_5_net2.item())

                ## Acc for net_mean.
                acc_1_net_mean, acc_5_net_mean = metrics.accuracy(outputs_mean, targets, topk=(1, 5))
                acc_val_top_1_net_mean.update(acc_1_net_mean.item())
                acc_val_top_5_net_mean.update(acc_5_net_mean.item())

        # Result dict.
        result_dict = {
            'epoch': epoch,
            'net1': {
                'loss_ce': loss_val_net1.get_avg(),
                'acc_top_1': acc_val_top_1_net1.get_avg(),
                'acc_top_5': acc_val_top_5_net1.get_avg()
            },
            'net2': {
                'loss_ce': loss_val_net2.get_avg(),
                'acc_top_1': acc_val_top_1_net2.get_avg(),
                'acc_top_5': acc_val_top_5_net2.get_avg()
            },
            'net_mean': {
                'acc_top_1': acc_val_top_1_net_mean.get_avg(),
                'acc_top_5': acc_val_top_5_net_mean.get_avg()
            }
        }
        if kwargs.get('test_data', False):
            self.history_info.setdefault('test_acc', []).append(result_dict['net_mean']['acc_top_1'])
            result_dict['avg_acc'] = funcs.get_average_value(self.history_info['test_acc'], 10)

        return result_dict
