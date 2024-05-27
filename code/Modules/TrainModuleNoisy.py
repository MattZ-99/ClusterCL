# -*- coding: utf-8 -*-
# @Time : 2022/10/16 9:28
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

from .TrainModuleClean import *


class TrainNoisyDataSingleNet(TrainCleanDataSingleNet):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataSingleNet, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_noisy_dataset()
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


class TrainNoisyDataSingleNetWithMixupAndCR(TrainCleanDataSingleNetWithMixupAndConsistencyRegularization):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataSingleNetWithMixupAndCR, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_noisy_weak_strong_dataset()
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_data = self.dataset_generator.generate_test_dataset()
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Training.
        for epoch in range(epochs):
            # Train one epoch.
            # training_info = self.train_basic_one_epoch(epoch, train_loader)
            training_info = self.train_one_epoch_with_CR_on_high_confident_data(epoch, train_loader)
            # test_info_train_data = self.test_basic_one_epoch(epoch, train_loader, CE_loss)
            test_info_test_data = self.test_basic_one_epoch(epoch, test_loader, test_data=True)

            scheduler.step()

            print("Train|Train data", training_info)
            # print("Test |Train data", test_info_train_data)
            print("Test |Test  data", test_info_test_data)
            print()

            self.update_logger_train(training_info)
            self.update_logger_test(test_info_test_data)


class TrainNoisyDataDualNetWithMixupAndCR(TrainCleanDataDualNetWithMixupAndCR):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetWithMixupAndCR, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def train(self, epochs):
        # Parameters.
        args = self.args
        scheduler = self.scheduler

        # Data loader.
        train_set = self.dataset_generator.generate_train_noisy_weak_strong_dataset()
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
