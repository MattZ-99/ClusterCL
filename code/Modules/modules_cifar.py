# -*- coding: utf-8 -*-
# @Time : 2022/7/15 20:27
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

import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from tqdm import tqdm
from tools import utils
import os


class TrainCifar:
    def __init__(self,
                 net: torch.nn.Module,
                 train_data_loader: torch.utils.data.Dataset,
                 test_data_loader: torch.utils.data.Dataset,
                 dataset_getter,
                 loss_fn,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 args: argparse.Namespace,
                 output_dir='./Output/'
                 ):
        self.dataset_getter = dataset_getter
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.net = net
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.args = args

        # Module paths
        self.output_dir = output_dir
        utils.makedirs(self.output_dir)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        utils.makedirs(self.checkpoint_dir)
        self.cluster_dir = os.path.join(self.output_dir, 'clusters')
        utils.makedirs(self.cluster_dir)

        self.save_args()

    def save_model(self, epoch=-1, path: str = None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')

        checkpoint = {
            'epoch': epoch,
            'stat_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)

        self.net.load_state_dict(checkpoint['stat_dict'])

    def save_args(self):
        utils.save_args(self.args, path=os.path.join(self.output_dir, 'args.txt'))

    def train_one_epoch(self, data_loader=None, optimizer=None):
        if data_loader is None:
            data_loader = self.train_data_loader

        if optimizer is None:
            optimizer = self.optimizer

        device = self.args.device
        net = self.net
        loss_fn = self.loss_fn

        net.train()
        loss_arr = []
        correctness_arr = []
        loop_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='[Train]')
        for iter_num, (data, target) in loop_bar:
            data, target = data.to(device), target.to(device)

            output = net(data)
            loss = loss_fn(output, target)

            _, predict = torch.max(output, dim=1)
            correctness = torch.eq(predict.cpu(), target.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correctness_arr.append(correctness)
            loss_arr.append(loss.item())

        correctness_arr = torch.cat(correctness_arr).float()
        return np.mean(np.array(loss_arr)), torch.mean(correctness_arr)

    def test_one_epoch(self, data_loader=None):
        if data_loader is None:
            data_loader = self.test_data_loader
        device = self.args.device
        net = self.net
        loss_fn = self.loss_fn

        net.eval()
        loss_arr = []
        correctness_arr = []
        with torch.no_grad():
            loop_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc='[Test] ')
            for iter_num, (data, target) in loop_bar:
                data, target = data.to(device), target.to(device)
                output = net(data)

                loss = loss_fn(output, target)

                _, predict = torch.max(output, dim=1)
                correctness = torch.eq(predict.cpu(), target.cpu())

                correctness_arr.append(correctness)
                loss_arr.append(loss.item())

            correctness_arr = torch.cat(correctness_arr).float()
        return np.mean(np.array(loss_arr)), torch.mean(correctness_arr)

    def train_cluster_one_epoch(self, data_loader, optimizer=None):
        net = self.net
        args = self.args
        net.train()

        if optimizer is None:
            optimizer = self.optimizer
        loss_fn = self.loss_fn

        loss_arr = []
        correctness_arr = []
        for batch_id, (data, label_true, label_noisy) in enumerate(data_loader):
            data, targets = data.to(args.device), label_noisy.to(args.device)
            outputs = net(data)

            _, predict = torch.max(outputs, dim=1)
            correctness = torch.eq(predict.cpu(), label_true.cpu())

            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correctness_arr.append(correctness)
            loss_arr.append(loss.item())

        correctness_arr = torch.cat(correctness_arr).float()
        return np.mean(np.array(loss_arr)), torch.mean(correctness_arr)

    def feature_extractor(self, data_loader):
        device = self.args.device
        net = self.net
        net.eval()

        feature_arr = []
        target_arr = []
        with torch.no_grad():
            feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
            for iter_num, (data, _, target) in enumerate(data_loader):
                data = data.to(device)
                features = feature_extractor(data)
                features = torch.flatten(features, 1).cpu()

                feature_arr.append(features)
                target_arr.append(target)

        feature_arr = torch.cat(feature_arr)
        target_arr = torch.cat(target_arr)
        return feature_arr, target_arr

    @staticmethod
    def feature_cluster_kmeans(data, cluster_num):
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, verbose=1)
        clusters = kmeans.fit_predict(data)

        return clusters

    @staticmethod
    def optimal_labels_for_clusters(clusters, label, cluster_num, label_num=10):
        cluster_optimal_label = []
        for i in range(cluster_num):
            index = (clusters == i)
            group_labels = label[index]
            group_labels_vector = np.zeros(label_num)
            for j in range(label_num):
                group_labels_vector[j] = np.sum((group_labels == j))

            group_labels_vector /= np.sum(group_labels_vector)
            label_optimal = np.argmax(group_labels_vector)
            cluster_optimal_label.append(label_optimal)
        cluster_optimal_label = torch.tensor(cluster_optimal_label).long()

        return cluster_optimal_label

    @staticmethod
    def cluster_for_optimal_label(data, label: torch.Tensor, cluster_num, label_num=10):
        label = label.numpy()
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, verbose=1)
        clusters = kmeans.fit_predict(data)
        cluster_optimal_label = []
        for i in range(cluster_num):
            index = (clusters == i)
            group_labels = label[index]
            group_labels_vector = np.zeros(label_num)
            for j in range(label_num):
                group_labels_vector[j] = np.sum((group_labels == j))

            group_labels_vector /= np.sum(group_labels_vector)
            label_optimal = np.argmax(group_labels_vector)
            cluster_optimal_label.append(label_optimal)
        cluster_optimal_label = torch.tensor(cluster_optimal_label).long()

        return clusters, cluster_optimal_label

    def train(self, epochs=10):
        log_train_path = os.path.join(self.output_dir, 'train.log')
        log_file = open(log_train_path, 'w')

        scheduler = self.scheduler

        dataset = self.dataset_getter.get_dataset_noise_only_base()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                                  num_workers=self.args.num_workers)

        for epoch in range(epochs + 1):
            train_loss, train_acc = self.train_one_epoch(data_loader=data_loader)
            print(f"[Train] Epoch: {epoch:0>4d}| loss: {train_loss:.4f}, accuracy:{train_acc:.4f}")

            test_loss, test_acc = self.test_one_epoch()
            print(f"[Test]  Epoch: {epoch:0>4d}| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

            log_train_str = '-' * 20 + '\n' \
                            + f"Epoch: {epoch:0>4d} \n" \
                            + f'train| loss: {train_loss:.4f}, accuracy:{train_acc:.4f} \n' \
                            + f'test | loss: {test_loss:.4f}, accuracy:{test_acc:.4f} \n' \
                            + '\n'
            log_file.write(log_train_str)
            log_file.flush()

            scheduler.step()

            if epoch % 50 == 0:
                self.save_model(epoch=epochs,
                                path=os.path.join(self.checkpoint_dir, f'epoch-{epoch}_test-acc-{test_acc:.4f}.pth'))

        log_file.close()

    def train_noisy(self, epochs=10):
        log_train_path = os.path.join(self.output_dir, 'train_noisy.log')
        log_file = open(log_train_path, 'w')

        args = self.args
        optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        dataset = self.dataset_getter.get_dataset_noise_only_base()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                                  num_workers=self.args.num_workers)

        dataset_clean = self.dataset_getter.get_dataset_clean_base()
        data_loader_clean_for_validation = torch.utils.data.DataLoader(dataset_clean,
                                                                       batch_size=self.args.batch_size, shuffle=False,
                                                                       num_workers=self.args.num_workers)

        for epoch in range(epochs):
            self.train_one_epoch(data_loader=data_loader, optimizer=optimizer)

            train_loss, train_acc = self.test_one_epoch(data_loader=data_loader_clean_for_validation)
            print(f"[Train] Epoch: {epoch:0>4d}| loss: {train_loss:.4f}, accuracy:{train_acc:.4f}")

            test_loss, test_acc = self.test_one_epoch()
            print(f"[Test]  Epoch: {epoch:0>4d}| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

            log_train_str = '-' * 20 + '\n' \
                            + f"Epoch: {epoch:0>4d} \n" \
                            + f'train| loss: {train_loss:.4f}, accuracy:{train_acc:.4f} \n' \
                            + f'test | loss: {test_loss:.4f}, accuracy:{test_acc:.4f} \n' \
                            + '\n'
            log_file.write(log_train_str)
            log_file.flush()

            scheduler.step()

            if epoch % 50 == 0:
                self.save_model(epoch=epochs,
                                path=os.path.join(self.checkpoint_dir, f'epoch-{epoch}_test-acc-{test_acc:.4f}.pth'))

        log_file.close()

        self.save_model(epoch=epochs, path=os.path.join(self.checkpoint_dir, f'model_warmup.pth'))

    def train_cluster(self, epochs=10, warmup=5):
        log_train_path = os.path.join(self.output_dir, 'train.log')
        log_file = open(log_train_path, 'w')

        scheduler = self.scheduler

        self.save_args()

        dataset = self.dataset_getter.get_dataset_noise_only_base()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                                                  num_workers=self.args.num_workers)
        for epoch in range(warmup):
            train_loss, train_acc = self.train_one_epoch(data_loader=data_loader)
            print(f"[Train] Epoch: {epoch:0>4d}| loss: {train_loss:.4f}, accuracy:{train_acc:.4f}")

            test_loss, test_acc = self.test_one_epoch()
            print(f"[Test]  Epoch: {epoch:0>4d}| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

            log_train_str = '-' * 20 + '\n' \
                            + f"Epoch: {epoch:0>4d} \n" \
                            + f'train| loss: {train_loss:.4f}, accuracy:{train_acc:.4f} \n' \
                            + f'test | loss: {test_loss:.4f}, accuracy:{test_acc:.4f} \n' \
                            + '\n'
            log_file.write(log_train_str)
            log_file.flush()

            scheduler.step()
        self.save_model(epoch=epochs, path=os.path.join(self.checkpoint_dir, f'model_warmup.pth'))

        dataset = self.dataset_getter.get_dataset_base()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                                  num_workers=self.args.num_workers)
        data_features, targets_arr = self.feature_extractor(data_loader=data_loader)
        clusters, cluster_optimal_label = self.cluster_for_optimal_label(data=data_features, label=targets_arr,
                                                                         cluster_num=1000)

        label_corrected = cluster_optimal_label[clusters]
        label_true, label_noisy = self.dataset_getter.get_label()
        print(f'Original noise rate:  {1 - np.mean(label_true == label_noisy)}')
        print(f'Corrected noise rate: {1 - np.mean(label_true == label_corrected.numpy())}')

        dataset_cluster = self.dataset_getter.get_dataset_cluster(clusters=clusters)
        data_loader = torch.utils.data.DataLoader(dataset_cluster, batch_size=self.args.batch_size, shuffle=True,
                                                  num_workers=self.args.num_workers)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_cluster_one_epoch(data_loader=data_loader,
                                                                 cluster_optimal_label=cluster_optimal_label)
            print(f"[Train] Epoch: {epoch:0>4d}| loss: {train_loss:.4f}, accuracy:{train_acc:.4f}")

            test_loss, test_acc = self.test_one_epoch()
            print(f"[Test]  Epoch: {epoch:0>4d}| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

            log_train_str = '-' * 20 + '\n' \
                            + f"Epoch: {epoch:0>4d} \n" \
                            + f'train| loss: {train_loss:.4f}, accuracy:{train_acc:.4f} \n' \
                            + f'test | loss: {test_loss:.4f}, accuracy:{test_acc:.4f} \n' \
                            + '\n'
            log_file.write(log_train_str)
            log_file.flush()

            scheduler.step()

        log_file.close()

    def train_cluster_v2(self, epochs=10, cluster_num=1000, cluster_interval: int = 10):
        log_train_path = os.path.join(self.output_dir, 'train_cluster.log')
        log_file = open(log_train_path, 'w')

        args = self.args

        optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        data_loader_corrected = None

        test_loss, test_acc = self.test_one_epoch()
        print(f"[Test]  Epoch: START| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

        for epoch in range(epochs):

            # Knn cluster for dataset
            if epoch % cluster_interval == 0:
                # dataset = self.dataset_getter.get_dataset_base()
                # data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                #                                           num_workers=self.args.num_workers)
                # data_features, targets_arr = self.feature_extractor(data_loader=data_loader)
                # clusters, cluster_optimal_label = self.cluster_for_optimal_label(data=data_features, label=targets_arr,
                #                                                                  cluster_num=cluster_num)
                cluster_result_dict = utils.load_pickle(
                    path='/home/mtzhang/projects/NoiseLabel/code/Outputs/2022_8_1/resnet18_pretrained_false_epochs-30_warmup-20_init-lr-0.01_seed-123_2022_8_1_19_3_57/clusters/epoch_0.pickle')
                clusters = cluster_result_dict['clusters']

                label_true, label_noisy = self.dataset_getter.get_label()
                cluster_optimal_label_true = self.optimal_labels_for_clusters(clusters, label_true,
                                                                              cluster_num=cluster_num)
                cluster_optimal_label_noisy = self.optimal_labels_for_clusters(clusters, label_noisy,
                                                                               cluster_num=cluster_num)

                print(self.dataset_getter.get_dataset_corrected(cluster_optimal_label_true[clusters].numpy()).noise_rate)

                # TODO: knn, 和 dividemix 结合/ multiple networks.
                #  batch cluster.

                # clusters = np.random.randint(0, cluster_num, size=50000)
                # cluster_optimal_label = torch.from_numpy(np.random.randint(0, 10, size=cluster_num))

                # print(clusters.shape, cluster_optimal_label.shape)
                # print(clusters)
                # print(cluster_optimal_label)

                # label correction
                cluster_optimal_label = cluster_optimal_label_noisy
                label_corrected = cluster_optimal_label[clusters].numpy()
                original_noise_rate = self.dataset_getter.noise_rate

                dataset_corrected = self.dataset_getter.get_dataset_corrected(label_corrected)
                data_loader_corrected = torch.utils.data.DataLoader(dataset_corrected, batch_size=self.args.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=self.args.num_workers)
                corrected_noise_rate = dataset_corrected.noise_rate

                print(f'Original noise rate:  {original_noise_rate}')
                print(f'Corrected noise rate: {corrected_noise_rate}')

                log_train_str = '\n' + f'Original noise rate:  {original_noise_rate}' + '\n' \
                                + f'Corrected noise rate: {corrected_noise_rate}' + '\n\n'
                log_file.write(log_train_str)
                log_file.flush()

                cluster_result_dict = {
                    'clusters': clusters,
                    'cluster_optimal_label': cluster_optimal_label,
                    'original_noise_rate': original_noise_rate,
                    'corrected_noise_rate': corrected_noise_rate
                }
                utils.save_pickle(cluster_result_dict,
                                  os.path.join(self.cluster_dir, f'epoch_{epoch}.pickle'))

            train_loss, train_acc = self.train_cluster_one_epoch(data_loader=data_loader_corrected, optimizer=optimizer)
            print(f"[Train] Epoch: {epoch:0>4d}| loss: {train_loss:.4f}, accuracy:{train_acc:.4f}")

            test_loss, test_acc = self.test_one_epoch()
            print(f"[Test]  Epoch: {epoch:0>4d}| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

            log_train_str = '-' * 20 + '\n' \
                            + f"Epoch: {epoch:0>4d} \n" \
                            + f'train| loss: {train_loss:.4f}, accuracy:{train_acc:.4f} \n' \
                            + f'test | loss: {test_loss:.4f}, accuracy:{test_acc:.4f} \n' \
                            + '\n'
            log_file.write(log_train_str)
            log_file.flush()

            scheduler.step()

        log_file.close()

    def train_knn(self, epochs=10, n_neighbors=2, cluster_interval: int = 10):
        log_train_path = os.path.join(self.output_dir, 'train_cluster.log')
        log_file = open(log_train_path, 'w')

        args = self.args

        optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        data_loader_corrected = None

        test_loss, test_acc = self.test_one_epoch()
        print(f"[Test]  Epoch: START| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

        for epoch in range(epochs):

            # Knn cluster for dataset
            if epoch % cluster_interval == 0:
                dataset = self.dataset_getter.get_dataset_base()
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False,
                                                          num_workers=self.args.num_workers)
                data_features, targets_arr = self.feature_extractor(data_loader=data_loader)

                # TODO: knn
                neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
                neigh.fit(data_features, targets_arr)
                prob = neigh.predict_proba(data_features)

                np.save(os.path.join(self.cluster_dir, f'epoch_{epoch}.npy'), prob)
                print(prob.shape)

                # label correction
                label_corrected = np.argmax(prob, axis=1)
                original_noise_rate = self.dataset_getter.noise_rate

                dataset_corrected = self.dataset_getter.get_dataset_corrected(label_corrected)
                data_loader_corrected = torch.utils.data.DataLoader(dataset_corrected, batch_size=self.args.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=self.args.num_workers)
                corrected_noise_rate = dataset_corrected.noise_rate

                print(f'Original noise rate:  {original_noise_rate}')
                print(f'Corrected noise rate: {corrected_noise_rate}')

                log_train_str = '\n' + f'Original noise rate:  {original_noise_rate}' + '\n' \
                                + f'Corrected noise rate: {corrected_noise_rate}' + '\n\n'
                log_file.write(log_train_str)
                log_file.flush()

            train_loss, train_acc = self.train_cluster_one_epoch(data_loader=data_loader_corrected, optimizer=optimizer)
            print(f"[Train] Epoch: {epoch:0>4d}| loss: {train_loss:.4f}, accuracy:{train_acc:.4f}")

            test_loss, test_acc = self.test_one_epoch()
            print(f"[Test]  Epoch: {epoch:0>4d}| loss: {test_loss:.4f}, accuracy:{test_acc:.4f}")

            log_train_str = '-' * 20 + '\n' \
                            + f"Epoch: {epoch:0>4d} \n" \
                            + f'train| loss: {train_loss:.4f}, accuracy:{train_acc:.4f} \n' \
                            + f'test | loss: {test_loss:.4f}, accuracy:{test_acc:.4f} \n' \
                            + '\n'
            log_file.write(log_train_str)
            log_file.flush()

            scheduler.step()

        log_file.close()

    def demo_run(self):
        self.train_one_epoch()
