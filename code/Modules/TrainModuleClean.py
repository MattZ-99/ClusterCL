# -*- coding: utf-8 -*-
# @Time : 2022/10/14 13:42
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
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from .BaseModules.BaseTrainingModule import *
from .tools import metrics, wrappers


class TrainCleanDataSingleNet(TrainSingleNetModule):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainCleanDataSingleNet, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

        self.scheduler = self.create_scheduler()

    def train_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        """ Train the network for one epoch.

        :param epoch: ...
        :param data_loader: Training data loader, batch format: (data, targets).
        :return: result_dict = {...}
        :rtype: dict
        """

        # Parameters.
        device = self.device
        net = self.net
        net.train()
        optimizer = self.optimizer
        loss_fn = self.loss_controller.get_cross_entropy_loss()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward.
            outputs = net(data)
            loss_ce = loss_fn(outputs, targets)

            # Backward.
            optimizer.zero_grad()
            loss_ce.backward()
            optimizer.step()

            # Log.
            loss_val.update(loss_ce.item())
            acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))

            acc_val_top_1.update(acc_1.item())
            acc_val_top_5.update(acc_5.item())

        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        result_dict = {
            'epoch': epoch,
            'loss_ce': loss_val.get_avg(),
            'lr': optimizer.param_groups[0]['lr'],
            'acc_top_1': acc_val_top_1.get_avg(),
            'acc_top_5': acc_val_top_5.get_avg()
        }

        return result_dict

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_clean_dataset()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data = self.dataset_generator.generate_test_dataset()
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Training.
        for epoch in range(epochs):
            # Train one epoch.
            training_info = self.train_basic_one_epoch(epoch, train_loader)
            # test_info_train_data = self.test_basic_one_epoch(epoch, train_loader, CE_loss)
            test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

            scheduler.step()

            print("Train|Train data", training_info)
            # print("Test |Train data", test_info_train_data)
            print("Test |Test  data", test_info_test_data)
            print()

            self.update_logger_train(training_info)
            self.update_logger_test(test_info_test_data)

    def create_scheduler(self):
        args = self.args
        optimizer = self.optimizer
        assert isinstance(optimizer, torch.optim.Optimizer)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 1e-5)

        return scheduler

    def update_logger_train(self, result_dict):
        output = f"|Epoch: {result_dict['epoch']:>4d}, Loss-CE: {result_dict['loss_ce']:.4f}, " \
                 f"Acc: {result_dict['acc_top_1']:.4f}, Acc(top-5): {result_dict['acc_top_5']:.4f}," \
                 f" LR: {result_dict['lr']:.8f}\n"

        self.logger_train.write(output)
        self.logger_train.flush()

    def update_logger_test(self, result_dict):
        output = f"|Epoch: {result_dict['epoch']:>4d}, Loss-CE: {result_dict['loss_ce']:.4f}, " \
                 f"Acc: {result_dict['acc_top_1']:.4f}, Acc(top-5): {result_dict['acc_top_5']:.4f}, " \
                 f"Average Acc (last 10 epochs): {result_dict['avg_acc']:.4f}\n"

        self.logger_test.write(output)
        self.logger_test.flush()


class TrainCleanDataSingleNetWithMixup(TrainCleanDataSingleNet):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainCleanDataSingleNetWithMixup, self).__init__(net, optimizer, loss_controller, dataset_generator,
                                                               device, args)

    def train_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        """ Train the network for one epoch.

        :param epoch: ...
        :param data_loader: Training data loader, batch format: (data, targets).
        :return: result_dict = {...}
        :rtype: dict
        """

        # Parameters.
        args = self.args
        device = self.device
        net = self.net
        net.train()
        optimizer = self.optimizer

        # loss functions.
        loss_fn = self.loss_controller.get_cross_entropy_loss()
        fmix_module = self.loss_controller.get_fmix_module()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            # target to one-hot
            targets = F.one_hot(targets, num_classes=args.num_classes).float()

            # CE loss.
            ## Forward.
            outputs = net(data)
            loss_ce = loss_fn(outputs, targets)

            # Mixup loss.
            ## Mixup.
            lamb_mix = np.random.beta(4, 4)
            lamb_mix = max(lamb_mix, 1 - lamb_mix)
            index = torch.randperm(data.size(0))
            data_mixed = lamb_mix * data + (1 - lamb_mix) * data[index, :]
            mixed_targets = lamb_mix * targets + (1 - lamb_mix) * targets[index]
            ## Forward.
            outputs_mixed = net(data_mixed)
            loss_mixup = loss_fn(outputs_mixed, mixed_targets)

            # Fmix loss.
            ## Fmix.
            data_fmix = fmix_module(data)
            ## Forward.
            outputs_fmix = net(data_fmix)
            loss_fmix = fmix_module.loss(outputs_fmix, targets)

            # Entire loss.
            loss = loss_ce + loss_mixup + loss_fmix
            # Backward.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target to class indices.
            targets = torch.argmax(targets, dim=1)

            # Log.
            loss_val.update(loss_ce.item())
            acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))

            acc_val_top_1.update(acc_1.item())
            acc_val_top_5.update(acc_5.item())

        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        result_dict = {
            'epoch': epoch,
            'loss_ce': loss_val.get_avg(),
            'lr': optimizer.param_groups[0]['lr'],
            'acc_top_1': acc_val_top_1.get_avg(),
            'acc_top_5': acc_val_top_5.get_avg()
        }

        return result_dict


class TrainCleanDataSingleNetWithMixupAndConsistencyRegularization(TrainCleanDataSingleNet):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainCleanDataSingleNetWithMixupAndConsistencyRegularization, self).__init__(net, optimizer,
                                                                                           loss_controller,
                                                                                           dataset_generator, device,
                                                                                           args)

    def train_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        """ Train the network for one epoch.

        :param epoch: ...
        :param data_loader: Training data loader, batch format: (data, targets).
        :return: result_dict = {...}
        :rtype: dict
        """

        # Parameters.
        args = self.args
        device = self.device
        net = self.net
        net.train()
        optimizer = self.optimizer

        # loss functions.
        loss_fn = self.loss_controller.get_cross_entropy_loss()
        fmix_module = self.loss_controller.get_fmix_module()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        for batch_idx, ([data, data_strong], targets) in enumerate(data_loader):
            data, data_strong, targets = data.to(device), data_strong.to(device), targets.to(device)
            # target to one-hot
            targets = F.one_hot(targets, num_classes=args.num_classes).float()

            # CE loss.
            ## Forward.
            outputs = net(data)
            loss_ce = loss_fn(outputs, targets)

            # Mixup loss.
            ## Mixup.
            lamb_mix = np.random.beta(4, 4)
            lamb_mix = max(lamb_mix, 1 - lamb_mix)
            index = torch.randperm(data.size(0))
            data_mixed = lamb_mix * data + (1 - lamb_mix) * data[index, :]
            mixed_targets = lamb_mix * targets + (1 - lamb_mix) * targets[index]
            ## Forward.
            outputs_mixed = net(data_mixed)
            loss_mixup = loss_fn(outputs_mixed, mixed_targets)

            # Fmix loss.
            ## Fmix.
            data_fmix = fmix_module(data)
            ## Forward.
            outputs_fmix = net(data_fmix)
            loss_fmix = fmix_module.loss(outputs_fmix, targets)

            # Consistency regularization loss.
            ## Forward.
            outputs_strong = net(data_strong)
            loss_cr = loss_fn(outputs_strong, targets)

            # Entire loss.
            loss = loss_ce + loss_mixup + loss_fmix + loss_cr
            # Backward.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target to class indices.
            targets = torch.argmax(targets, dim=1)

            # Log.
            loss_val.update(loss_ce.item())
            acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))

            acc_val_top_1.update(acc_1.item())
            acc_val_top_5.update(acc_5.item())

        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        result_dict = {
            'epoch': epoch,
            'loss_ce': loss_val.get_avg(),
            'lr': optimizer.param_groups[0]['lr'],
            'acc_top_1': acc_val_top_1.get_avg(),
            'acc_top_5': acc_val_top_5.get_avg()
        }

        return result_dict

    def train_one_epoch_with_CR_on_high_confident_data(self, epoch: int, data_loader, *args, **kwargs):
        """ Train the network for one epoch.

        :param epoch: ...
        :param data_loader: Training data loader, batch format: (data, targets).
        :return: result_dict = {...}
        :rtype: dict
        """

        # Parameters.
        args = self.args
        device = self.device
        net = self.net
        net.train()
        optimizer = self.optimizer

        # loss functions.
        loss_fn = self.loss_controller.get_cross_entropy_loss()
        fmix_module = self.loss_controller.get_fmix_module()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        for batch_idx, ([data, data_strong], targets) in enumerate(data_loader):
            data, data_strong, targets = data.to(device), data_strong.to(device), targets.to(device)
            # target to one-hot
            targets = F.one_hot(targets, num_classes=args.num_classes).float()

            # CE loss.
            ## Forward.
            outputs = net(data)
            loss_ce = loss_fn(outputs, targets)

            # Mixup loss.
            ## Mixup.
            lamb_mix = np.random.beta(4, 4)
            lamb_mix = max(lamb_mix, 1 - lamb_mix)
            index = torch.randperm(data.size(0))
            data_mixed = lamb_mix * data + (1 - lamb_mix) * data[index, :]
            mixed_targets = lamb_mix * targets + (1 - lamb_mix) * targets[index]
            ## Forward.
            outputs_mixed = net(data_mixed)
            loss_mixup = loss_fn(outputs_mixed, mixed_targets)

            # Fmix loss.
            ## Fmix.
            data_fmix = fmix_module(data)
            ## Forward.
            outputs_fmix = net(data_fmix)
            loss_fmix = fmix_module.loss(outputs_fmix, targets)

            # Consistency regularization loss.
            ## Calculate targets and weights for CR.
            with torch.no_grad():
                prob_weak = F.softmax(outputs.detach(), dim=1)
                prob_max = torch.max(prob_weak, dim=1)
                weights_cr = prob_max[0].gt(args.tau).float()
                targets_strong = prob_max[1].long()

            ## Forward.
            if torch.sum(weights_cr) == 0:
                loss_cr = torch.tensor(0.0).to(device)
            else:
                outputs_strong = net(data_strong)
                loss_cr = torch.sum(F.cross_entropy(outputs_strong, targets_strong, reduction='none') * weights_cr)
                loss_cr /= torch.sum(weights_cr)

            # Entire loss.
            loss = loss_ce + loss_mixup + loss_fmix + loss_cr
            # Backward.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target to class indices.
            targets = torch.argmax(targets, dim=1)

            # Log.
            loss_val.update(loss_ce.item())
            acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))

            acc_val_top_1.update(acc_1.item())
            acc_val_top_5.update(acc_5.item())

        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        result_dict = {
            'epoch': epoch,
            'loss_ce': loss_val.get_avg(),
            'lr': optimizer.param_groups[0]['lr'],
            'acc_top_1': acc_val_top_1.get_avg(),
            'acc_top_5': acc_val_top_5.get_avg()
        }

        return result_dict

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_clean_weak_strong_dataset()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data = self.dataset_generator.generate_test_dataset()
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Training.
        for epoch in range(epochs):
            # Train one epoch.
            training_info = self.train_basic_one_epoch(epoch, train_loader)
            # test_info_train_data = self.test_basic_one_epoch(epoch, train_loader, CE_loss)
            test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

            scheduler.step()

            print("Train|Train data", training_info)
            # print("Test |Train data", test_info_train_data)
            print("Test |Test  data", test_info_test_data)
            print()

            self.update_logger_train(training_info)
            self.update_logger_test(test_info_test_data)

            # Save model.
            if args.save_model and epoch == epochs - 1 or epoch % args.save_interval == 0:
                self.save_model(epoch=epoch)


class TrainCleanDataDualNet(TrainDualNetModule):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainCleanDataDualNet, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

        self.scheduler = self.create_scheduler()

    def train_basic_one_epoch_for_one_net(self, epoch: int, data_loader, net, optimizer, *args, **kwargs):
        """ Train the network for one epoch.

        :param optimizer:
        :param net:
        :param epoch: ...
        :param data_loader: Training data loader, batch format: (data, targets).
        :return: result_dict = {...}
        :rtype: dict
        """

        # Parameters.
        device = self.device
        net.train()
        loss_fn = self.loss_controller.get_cross_entropy_loss()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        for batch_idx, ((data, _), targets, _) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward.
            outputs = net(data)
            loss_ce = loss_fn(outputs, targets)

            # Backward.
            optimizer.zero_grad()
            loss_ce.backward()
            optimizer.step()

            # Log.
            loss_val.update(loss_ce.item())
            acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))

            acc_val_top_1.update(acc_1.item())
            acc_val_top_5.update(acc_5.item())

        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        result_dict = {
            'loss_ce': loss_val.get_avg(),
            'lr': optimizer.param_groups[0]['lr'],
            'acc_top_1': acc_val_top_1.get_avg(),
            'acc_top_5': acc_val_top_5.get_avg()
        }

        return result_dict

    def train_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        training_info_1 = self.train_basic_one_epoch_for_one_net(epoch, data_loader, self.net.net1, self.optimizer[0])
        training_info_2 = self.train_basic_one_epoch_for_one_net(epoch, data_loader, self.net.net2, self.optimizer[1])

        training_info = {
            'epoch': epoch,
            'net1': training_info_1,
            'net2': training_info_2,
            'lr': (self.optimizer[0].param_groups[0]['lr'], self.optimizer[1].param_groups[0]['lr'])
        }

        return training_info

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_clean_dataset()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data = self.dataset_generator.generate_test_dataset()
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Training.
        for epoch in range(epochs):
            # Train one epoch.
            training_info = self.train_basic_one_epoch(epoch, train_loader)
            # test_info_train_data = self.test_basic_one_epoch(epoch, train_loader, CE_loss)
            test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

            scheduler.step()

            print("Train|Train data", training_info)
            # print("Test |Train data", test_info_train_data)
            print("Test |Test  data", test_info_test_data)
            print()

            self.update_logger_train(training_info)
            self.update_logger_test(test_info_test_data)

            # Save model.
            if args.save_model and epoch == epochs - 1:
                self.save_model(epoch=epoch)

    @staticmethod
    def _get_scheduler(optimizer, scheduler_mode, args):
        if scheduler_mode == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-5)
        elif scheduler_mode == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs / 2, gamma=0.1)
        else:
            raise ValueError('Unknown scheduler mode: {}'.format(scheduler_mode))

    def create_scheduler(self):
        args = self.args
        optimizer_list = self.optimizer

        scheduler_mode = vars(args).get('scheduler_mode', 'cosine')

        scheduler_list = []
        for optimizer in optimizer_list:
            scheduler = self._get_scheduler(optimizer, scheduler_mode, args)
            scheduler_list.append(scheduler)

        scheduler_wrapper = wrappers.SchedulerWrapper(scheduler_list)
        return scheduler_wrapper

    def update_logger_train(self, result_dict):
        output = f"|Epoch: {result_dict['epoch']:>4d}\n"
        output += ' ' * 14 + f"-net1- Acc: {result_dict['net1']['acc_top_1']:.4f}, Loss-CE: {result_dict['net1']['loss_ce']:.4f}, LR: {result_dict['net1']['lr']:.8f} \n"
        output += ' ' * 14 + f"-net2- Acc: {result_dict['net2']['acc_top_1']:.4f}, Loss-CE: {result_dict['net2']['loss_ce']:.4f}, LR: {result_dict['net2']['lr']:.8f}  \n"

        self.logger_train.write(output)
        self.logger_train.flush()

    def update_logger_test(self, result_dict):
        output = f"|Epoch: {result_dict['epoch']:>4d}, Acc-mean: {result_dict['net_mean']['acc_top_1']:.4f}, Average Acc (last 10 epochs): {result_dict['avg_acc']:.4f} \n"
        output += ' ' * 14 + f"-net1- Acc: {result_dict['net1']['acc_top_1']:.4f}, Loss-CE: {result_dict['net1']['loss_ce']:.4f} \n"
        output += ' ' * 14 + f"-net2- Acc: {result_dict['net2']['acc_top_1']:.4f}, Loss-CE: {result_dict['net2']['loss_ce']:.4f} \n"

        self.logger_test.write(output)
        self.logger_test.flush()


class TrainCleanDataDualNetWithMixupAndCR(TrainCleanDataDualNet):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainCleanDataDualNetWithMixupAndCR, self).__init__(net, optimizer, loss_controller, dataset_generator,
                                                                  device, args)

    def loss_ce_mixup_cr_for_one_net(self, net, data, data_strong, targets):
        # loss functions.
        loss_fn = self.loss_controller.get_cross_entropy_loss()
        fmix_module = self.loss_controller.get_fmix_module()

        # CE loss.
        ## Forward.
        outputs = net(data)
        loss_ce = loss_fn(outputs, targets)

        # Mixup loss.
        ## Mixup.
        lamb_mix = np.random.beta(4, 4)
        lamb_mix = max(lamb_mix, 1 - lamb_mix)
        index = torch.randperm(data.size(0))
        data_mixed = lamb_mix * data + (1 - lamb_mix) * data[index, :]
        mixed_targets = lamb_mix * targets + (1 - lamb_mix) * targets[index]
        ## Forward.
        outputs_mixed = net(data_mixed)
        loss_mixup = loss_fn(outputs_mixed, mixed_targets)

        # Fmix loss.
        ## Fmix.
        data_fmix = fmix_module(data)
        ## Forward.
        outputs_fmix = net(data_fmix)
        loss_fmix = fmix_module.loss(outputs_fmix, targets)

        # Consistency regularization loss.
        ## Forward.
        outputs_strong = net(data_strong)
        loss_cr = loss_fn(outputs_strong, targets)

        # Consistency regularization loss.
        ## Forward.
        outputs_strong = net(data_strong)
        loss_cr = loss_fn(outputs_strong, targets)

        return outputs, (loss_ce, loss_mixup, loss_fmix, loss_cr)

    def train_basic_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        # Parameters.
        device = self.device
        args = self.args
        loss_calculator = self.loss_ce_mixup_cr_for_one_net

        ## Nets.
        net1 = self.net.net1
        net2 = self.net.net2
        net1.train()
        net2.train()

        ## Optimizers.
        optimizer1 = self.optimizer[0]
        optimizer2 = self.optimizer[1]

        # [[loss_ce, acc_top_1, acc_top_5] for net1, [loss_ce, acc_top_1, acc_top_5] for net2]
        vals_stat = utils.ValueListStat(shape=(2, 3))

        for batch_idx, ([data, data_strong], targets) in enumerate(data_loader):
            data, data_strong, targets = data.to(device), data_strong.to(device), targets.to(device)
            # target to one-hot
            targets = F.one_hot(targets, num_classes=args.num_classes).float()

            # ----------------- Train net1 ----------------- #
            outputs_1, (loss_ce_1, loss_mixup_1, loss_fmix_1, loss_cr_1) = loss_calculator(net1, data, data_strong,
                                                                                           targets)

            # Entire loss.
            loss_1 = loss_ce_1 + loss_mixup_1 + loss_fmix_1 + loss_cr_1

            # Backward.
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

            # ----------------- Train net2 ----------------- #
            outputs_2, (loss_ce_2, loss_mixup_2, loss_fmix_2, loss_cr_2) = loss_calculator(net2, data, data_strong,
                                                                                           targets)

            # Entire loss.
            loss_2 = loss_ce_2 + loss_mixup_2 + loss_fmix_2 + loss_cr_2

            # Backward.
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            # target to class indices.
            targets = torch.argmax(targets, dim=1)

            # Log.
            acc_1_net1, acc_5_net1 = metrics.accuracy(outputs_1, targets, topk=(1, 5))
            acc_1_net2, acc_5_net2 = metrics.accuracy(outputs_2, targets, topk=(1, 5))
            vals_stat.update([[loss_ce_1.item(), acc_1_net1.item(), acc_5_net1.item()],
                              [loss_ce_2.item(), acc_1_net2.item(), acc_5_net2.item()]])

        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        vals_avg = vals_stat.get_avg()

        result_dict = {
            'epoch': epoch,
            'net1': {
                'loss_ce': vals_avg[0][0],
                'acc_top_1': vals_avg[0][1],
                'acc_top_5': vals_avg[0][2],
                'lr': optimizer1.param_groups[0]['lr']
            },
            'net2': {
                'loss_ce': vals_avg[1][0],
                'acc_top_1': vals_avg[1][1],
                'acc_top_5': vals_avg[1][2],
                'lr': optimizer2.param_groups[0]['lr']
            },
            'lr': (optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'])
        }

        return result_dict

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_clean_weak_strong_dataset()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data = self.dataset_generator.generate_test_dataset()
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Training.
        for epoch in range(epochs):
            # Train one epoch.
            training_info = self.train_basic_one_epoch(epoch, train_loader)
            # test_info_train_data = self.test_basic_one_epoch(epoch, train_loader, CE_loss)
            test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

            scheduler.step()

            print("Train|Train data", training_info)
            # print("Test |Train data", test_info_train_data)
            print("Test |Test  data", test_info_test_data)
            print()

            self.update_logger_train(training_info)
            self.update_logger_test(test_info_test_data)

            # Save model.
            if args.save_model and epoch == epochs - 1 or epoch % args.save_interval == 0:
                self.save_model(epoch=epoch)
