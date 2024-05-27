# -*- coding: utf-8 -*-
# @Time : 2022/10/17 17:29
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

from .TrainModuleNoisy import *


class TrainNoisyDataDualNetWithWarmup(TrainCleanDataDualNet):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetWithWarmup, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def warmup_one_epoch_for_one_net(self, epoch: int, data_loader, net, optimizer, *args, **kwargs):
        # Parameters.
        device = self.device
        net.train()

        loss_fn = self.loss_controller.get_cross_entropy_loss()
        conf_penalty = self.loss_controller.get_neg_entropy()

        # [loss_ce, acc_top_1, acc_top_5]
        vals_stat = utils.ValueListStat(shape=(3,))
        for batch_idx, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward.
            outputs = net(data)
            loss_ce = loss_fn(outputs, targets)

            # Loss penalty with negative entropy.
            penalty = conf_penalty(outputs)

            loss = loss_ce + penalty

            # Backward.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log.
            acc_1, acc_5 = metrics.accuracy(outputs, targets, topk=(1, 5))
            vals_stat.update([loss_ce.item(), acc_1.item(), acc_5.item()])
        # Result dict.
        ## Acc is calculated by the training targets, which may be noisy.
        vals_avg = vals_stat.get_avg()
        result_dict = {
            'loss_ce': vals_avg[0],
            'lr': optimizer.param_groups[0]['lr'],
            'acc_top_1': vals_avg[1],
            'acc_top_5': vals_avg[2]
        }

        return result_dict

    def warmup_one_epoch(self, epoch: int, data_loader, *args, **kwargs):
        if kwargs.get("warmup_net1", True):
            training_info_1 = self.warmup_one_epoch_for_one_net(epoch, data_loader, self.net.net1, self.optimizer[0])
        else:
            training_info_1 = None
        if kwargs.get("warmup_net2", True):
            training_info_2 = self.warmup_one_epoch_for_one_net(epoch, data_loader, self.net.net2, self.optimizer[1])
        else:
            training_info_2 = None

        training_info = {
            'epoch': epoch,
            'net1': training_info_1,
            'net2': training_info_2,
            'lr': (self.optimizer[0].param_groups[0]['lr'], self.optimizer[1].param_groups[0]['lr'])
        }

        return training_info

    def train_main_one_epoch(self, epoch: int, *args, **kwargs):
        raise NotImplementedError

    def train_with_warmup(self, epochs, epochs_warmup):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        warmup_set = self.dataset_generator.generate_train_noisy_dataset()
        warmup_loader = DataLoader(warmup_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data = self.dataset_generator.generate_test_dataset()
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for epoch in range(epochs):
            if epoch < epochs_warmup:
                # Warmup.
                training_info = self.warmup_one_epoch(epoch, warmup_loader)

                # if epoch < 2:
                #     training_info = self.warmup_one_epoch(epoch, warmup_loader)
                # else:
                #     self.warmup_one_epoch(epoch, warmup_loader, warmup_net1=False)
                #     training_info = None

                test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

                self.update_logger_train_with_name(training_info, "Warmup")
                self.update_logger_test_with_name(test_info_test_data, "Warmup")

            else:
                # Main training method.
                training_info = self.train_main_one_epoch(epoch)
                test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

                self.update_logger_train_with_name(training_info, "Main")
                self.update_logger_test_with_name(test_info_test_data, "Main")

            print("Train|Train data", training_info)
            print("Test |Test  data", test_info_test_data)
            print()

            scheduler.step()

            # Save model.
            if args.save_model and (epoch == epochs - 1 or epoch % args.save_interval == 0):
                self.save_model(epoch=epoch)

        # Save final model.
        self.save_model(epoch='final')

    def update_logger_train_with_name(self, result_dict, name: str):
        if result_dict is None:
            return

        output = f"|{name}-Epoch: {result_dict['epoch']:>4d}\n"
        output += ' ' * (15 + len(name)) + f"-net1- Acc: {result_dict['net1']['acc_top_1']:.4f}, Loss-CE: {result_dict['net1']['loss_ce']:.4f}, LR: {result_dict['net1']['lr']:.8f} \n"
        output += ' ' * (15 + len(name)) + f"-net2- Acc: {result_dict['net2']['acc_top_1']:.4f}, Loss-CE: {result_dict['net2']['loss_ce']:.4f}, LR: {result_dict['net2']['lr']:.8f}  \n"

        self.logger_train.write(output)
        self.logger_train.flush()

    def update_logger_test_with_name(self, result_dict, name: str):
        output = f"|{name}-Epoch: {result_dict['epoch']:>4d}, Acc-mean: {result_dict['net_mean']['acc_top_1']:.4f}, Average Acc (last 10 epochs): {result_dict['avg_acc']:.4f} \n"
        output += ' ' * (15 + len(name)) + f"-net1- Acc: {result_dict['net1']['acc_top_1']:.4f}, Loss-CE: {result_dict['net1']['loss_ce']:.4f} \n"
        output += ' ' * (15 + len(name)) + f"-net2- Acc: {result_dict['net2']['acc_top_1']:.4f}, Loss-CE: {result_dict['net2']['loss_ce']:.4f} \n"

        self.logger_test.write(output)
        self.logger_test.flush()
