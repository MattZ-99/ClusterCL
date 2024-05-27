# -*- coding: utf-8 -*-
# @Time : 2022/8/21 17:00
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
import math
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torchvision.transforms import transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tqdm import tqdm
from .funcs import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from tools import utils
from repositories.NMCE.NMCE.architectures.models import Gumble_Softmax
from repositories.NMCE.NMCE.func import cluster_match, cluster_merge_match, normalized_mutual_info_score
from repositories.NMCE.NMCE.func import adjusted_rand_score, save_latent_pca_figure, save_cluster_imgs, warmup_lr
from repositories.NMCE.NMCE.loss import TotalCodingRate, Z_loss, MaximalCodingRateReduction
from repositories.NMCE.NMCE.lars import LARS, LARSWrapper


class DivideMixTrainCifar:
    def __init__(self, net_1=None, net_2=None, optimizer_1=None, optimizer_2=None,
                 dataset_getter=None, loss_controller=None,
                 device=None, args=None):
        super(DivideMixTrainCifar, self).__init__()
        self.loss_controller = loss_controller
        self.dataset_getter = dataset_getter
        self.device = device
        self.optimizer_2 = optimizer_2
        self.optimizer_1 = optimizer_1
        self.net_2 = net_2
        self.net_1 = net_1
        self.args = args

        # Elements.
        self.info_dict = {
            'test_acc': [],
        }

        # Path
        self.root_dir = args.output_dir
        self.checkpoint_dir = os.path.join(self.root_dir, 'checkpoints')
        utils.makedirs(self.checkpoint_dir)

    def save_model(self, epoch=-1, path: str = None):
        if path is None:
            path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')

        checkpoint_net_1 = {
            'epoch': epoch,
            'stat_dict': self.net_1.state_dict(),
            'optimizer': self.optimizer_1.state_dict()
        }
        checkpoint_net_2 = {
            'epoch': epoch,
            'stat_dict': self.net_2.state_dict(),
            'optimizer': self.optimizer_2.state_dict()
        }

        checkpoint = {
            'net_1': checkpoint_net_1,
            'net_2': checkpoint_net_2
        }
        torch.save(checkpoint, path)

    # def load_model(self, path: str):
    #     checkpoint = torch.load(path)
    #     self.net_1.load_state_dict(checkpoint['net_1']['stat_dict'])
    #     self.net_2.load_state_dict(checkpoint['net_2']['stat_dict'])
    #     self.optimizer_1.load_state_dict(checkpoint['net_1']['optimizer'])
    #     self.optimizer_2.load_state_dict(checkpoint['net_2']['optimizer'])

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.net_1.backbone.load_state_dict(checkpoint['net_1']['stat_dict'], strict=False)
        self.net_2.backbone.load_state_dict(checkpoint['net_2']['stat_dict'], strict=False)
        self.net_1.load_state_dict(checkpoint['net_1']['stat_dict'], strict=False)
        self.net_2.load_state_dict(checkpoint['net_2']['stat_dict'], strict=False)

    def test(self, epoch: int, net_1, net_2, test_loader):
        # Parameters
        args = self.args
        device = self.device

        net_1.eval()
        net_2.eval()

        acc_val = utils.ValueStat()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net_1(inputs)
                outputs2 = net_2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                acc_val.update(torch.mean(predicted.eq(targets).float()).item())

        print(f"| Test Epoch #{epoch}\t Accuracy: {100. * acc_val.get_avg():.2f}%\n\n")
        # test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
        # test_log.flush()
        return acc_val.get_avg()

    def test_single_net(self, epoch: int, net, test_loader):
        # Parameters
        args = self.args
        device = self.device

        net.eval()

        acc_val = utils.ValueStat()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)

                acc_val.update(torch.mean(predicted.eq(targets).float()).item())

        print(f"| Test Epoch #{epoch}\t Accuracy: {100. * acc_val.get_avg():.2f}%\n\n")
        return acc_val.get_avg()

    def warmup_one_epoch(self, epoch, net, optimizer, data_loader):
        # Parameters
        args = self.args
        loss_controller = self.loss_controller
        device = self.device

        ce_loss = loss_controller.cross_entropy_loss
        conf_penalty = loss_controller.conf_penalty

        num_iter = (len(data_loader.dataset) // data_loader.batch_size) + 1
        loss_val = utils.ValueStat()

        net.train()

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = ce_loss(outputs, labels)

            penalty = conf_penalty(outputs)
            L = loss + penalty

            L.backward()
            optimizer.step()

            loss_val.update(loss.item())

            sys.stdout.write('\r')
            sys.stdout.write(f'{args.dataset}-{args.noise_mode} |'
                             f' Epoch [{epoch:>3d}/{args.epochs:>3d}] Iter[{batch_idx + 1:>3d}/{num_iter:>3d}]'
                             f'\t CE-loss: {loss_val.get_avg():.4f}'
                             )
            sys.stdout.flush()

    def train_base_one_epoch(self, epoch, net, optimizer, data_loader):
        # Parameters
        args = self.args
        loss_controller = self.loss_controller
        device = self.device

        ce_loss = loss_controller.cross_entropy_loss

        num_iter = (len(data_loader.dataset) // data_loader.batch_size) + 1
        loss_val = utils.ValueStat()

        net.train()

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = ce_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_val.update(loss.item())

            sys.stdout.write('\r')
            sys.stdout.write(f'{args.dataset}-{args.noise_mode} |'
                             f' Epoch [{epoch:>3d}/{args.epochs:>3d}] Iter[{batch_idx + 1:>3d}/{num_iter:>3d}]'
                             f'\t CE-loss: {loss_val.get_avg():.4f}'
                             )
            sys.stdout.flush()

    def eval_prob_clean_for_dataset_division(self, net, data_loader):
        # Parameters
        args = self.args
        device = self.device

        ce_loss_none = self.loss_controller.cross_entropy_none

        net.eval()
        if args.dataset == 'cifar10':
            losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 10))
        elif args.dataset == 'cifar100':
            losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 100))
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = ce_loss_none(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]
        losses = (losses - losses.min()) / (losses.max() - losses.min()).cpu().numpy()

        # fit a two-component GMM to the loss
        input_loss = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]

        return prob

    def train_semi_supervised_one_epoch(self, epoch, net_1, net_2, optimizer, labeled_loader, unlabeled_loader):
        # Parameters
        args = self.args
        device = self.device
        criterion = self.loss_controller.semi_loss

        net_1.train()
        net_2.eval()  # fix one network and train the other

        unlabeled_train_iter = iter(unlabeled_loader)
        num_iter = (len(labeled_loader.dataset) // args.batch_size) + 1

        for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_loader):
            try:
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_loader)
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            batch_size = len(inputs_x)

            # Transform label to one-hot
            labels_x = F.one_hot(labels_x, num_classes=args.num_class)
            w_x = w_x.view(-1, 1).float()

            inputs_x, inputs_x2 = inputs_x.to(device), inputs_x2.to(device)
            labels_x, w_x = labels_x.to(device), w_x.to(device)
            inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net_1(inputs_u)
                outputs_u12 = net_1(inputs_u2)
                outputs_u21 = net_2(inputs_u)
                outputs_u22 = net_2(inputs_u2)

                pu = (
                             torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                             torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)
                     ) / 4
                ptu = pu ** (1 / args.T)  # temperature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x = net_1(inputs_x)
                outputs_x2 = net_1(inputs_x2)

                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)  # temperature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

                # targets_x = labels_x.detach()

            # mixmatch
            lamb_mix = np.random.beta(args.alpha, args.alpha)
            lamb_mix = max(lamb_mix, 1 - lamb_mix)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = lamb_mix * input_a + (1 - lamb_mix) * input_b
            mixed_target = lamb_mix * target_a + (1 - lamb_mix) * target_b

            losses = net_1(mixed_input)
            losses_x = losses[:batch_size * 2]
            losses_u = losses[batch_size * 2:]

            Lx, Lu, lamb = criterion(losses_x, mixed_target[:batch_size * 2], losses_u, mixed_target[batch_size * 2:],
                                     epoch + batch_idx / num_iter, args.epochs_warmup)

            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.to(device)
            pred_mean = torch.softmax(losses, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss_all = Lx + lamb * Lu + penalty

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write(f'{args.dataset}-{args.noise_mode} | Epoch [{epoch:>3d}/{args.epochs:>3d}]'
                             f' Iter[{batch_idx + 1:>3d}/{num_iter:>3d}]\t'
                             f' Labeled loss: {Lx.item():.2f}  Unlabeled loss: {Lu.item():.4f}'
                             )
            sys.stdout.flush()

    def train(self, epochs, epochs_warmup):
        # Parameters
        optimizer_1 = self.optimizer_1
        optimizer_2 = self.optimizer_2
        net_1 = self.net_1
        net_2 = self.net_2

        args = self.args
        lr = args.lr

        # Constructor.
        log_acc_file = open(os.path.join(self.root_dir, 'acc.log'), 'w')

        # Data loader.
        warmup_dataset = self.dataset_getter.get_warmup_dataset()
        warmup_loader = torch.utils.data.DataLoader(warmup_dataset, batch_size=2 * args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, drop_last=False)
        eval_dataset = self.dataset_getter.get_eval_dataset()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)
        test_dataset = self.dataset_getter.get_test_dataset()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        for epoch in range(epochs + 1):

            # learning rate schedule
            if epoch == epochs / 2:
                lr /= 10
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr

            # Train warm up.
            if epoch < epochs_warmup:
                print('Warmup Net1')
                self.warmup_one_epoch(epoch, net_1, optimizer_1, warmup_loader)
                print('\nWarmup Net2')
                self.warmup_one_epoch(epoch, net_2, optimizer_2, warmup_loader)
                print('\n' * 2)

            # Train dividemix.
            else:
                prob_1 = self.eval_prob_clean_for_dataset_division(net_1, eval_loader)
                prob_2 = self.eval_prob_clean_for_dataset_division(net_2, eval_loader)
                pred_1 = np.array(prob_1 > args.p_threshold, dtype=bool)
                pred_2 = np.array(prob_2 > args.p_threshold, dtype=bool)

                print(f"Labeled data has a size of {pred_1.sum()}, {pred_2.sum()}")
                print(f"Unlabeled data has a size of {np.logical_not(pred_1).sum()}, {np.logical_not(pred_2).sum()}")

                print('Train Net1')
                labeled_dataset, unlabeled_dataset = self.dataset_getter.get_divide_datasets(pred=pred_2, prob=prob_2)
                labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_workers, drop_last=False)
                unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                               shuffle=True, num_workers=args.num_workers,
                                                               drop_last=False)

                self.train_semi_supervised_one_epoch(epoch, net_1, net_2, optimizer_1, labeled_loader, unlabeled_loader)

                print('\nTrain Net2')
                labeled_dataset, unlabeled_dataset = self.dataset_getter.get_divide_datasets(pred=pred_1, prob=prob_1)
                labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_workers, drop_last=False)
                unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                               shuffle=True, num_workers=args.num_workers,
                                                               drop_last=False)
                self.train_semi_supervised_one_epoch(epoch, net_2, net_1, optimizer_2, labeled_loader, unlabeled_loader)

                print('\n' * 2)

            # Test.
            test_acc = self.test(epoch, net_1, net_2, test_loader)
            # Log.
            self.info_dict['test_acc'].append(test_acc)

            log_acc_file.write(f'epoch={epoch:>3d}, accuracy={test_acc:.4f},'
                               f" average accuracy(last 10 epochs)={get_average_value(self.info_dict['test_acc'], last_n=10):.4f}\n")  # log accuracy.
            log_acc_file.flush()

            # Save model.
            if epoch % 1 == 0:
                self.save_model(epoch,
                                path=os.path.join(self.checkpoint_dir, f'epoch-{epoch}_test-acc-{test_acc:.4f}.pth'))

        # Destructor.
        log_acc_file.close()

    def train_noisy(self, epochs):
        # Parameters
        optimizer_1 = self.optimizer_1
        net_1 = self.net_1

        args = self.args
        lr = args.lr

        # Constructor.
        log_acc_file = open(os.path.join(self.root_dir, 'acc.log'), 'w')

        # Data loader.
        warmup_dataset = self.dataset_getter.get_warmup_dataset()
        warmup_loader = torch.utils.data.DataLoader(warmup_dataset, batch_size=2 * args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, drop_last=False)
        test_dataset = self.dataset_getter.get_test_dataset()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        for epoch in range(epochs + 1):

            # learning rate schedule
            adjust_learning_rate(optimizer_1, epoch, epochs + 1, args.lr, cosine=True, lr_decay_rate=0.1)

            print('Warmup Net1')
            self.train_base_one_epoch(epoch, net_1, optimizer_1, warmup_loader)

            # Test.
            test_acc = self.test_single_net(epoch, net_1, test_loader)
            # Log.
            self.info_dict['test_acc'].append(test_acc)

            log_acc_file.write(f'epoch={epoch:>3d}, accuracy={test_acc:.4f},'
                               f" average accuracy(last 10 epochs)={get_average_value(self.info_dict['test_acc'], last_n=10):.4f}\n")  # log accuracy.
            log_acc_file.flush()

            # Save model.
            if epoch % 1 == 0:
                self.save_model(epoch,
                                path=os.path.join(self.checkpoint_dir, f'epoch-{epoch}_test-acc-{test_acc:.4f}.pth'))

        # Destructor.
        log_acc_file.close()

    def train_clean(self, epochs):
        # Parameters
        optimizer_1 = self.optimizer_1
        net_1 = self.net_1

        args = self.args
        lr = args.lr

        # Constructor.
        log_acc_file = open(os.path.join(self.root_dir, 'acc.log'), 'w')

        # Data loader.
        warmup_dataset = self.dataset_getter.get_train_clean_dataset()
        warmup_loader = torch.utils.data.DataLoader(warmup_dataset, batch_size=2 * args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, drop_last=False)
        test_dataset = self.dataset_getter.get_test_dataset()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        for epoch in range(epochs + 1):

            # learning rate schedule
            adjust_learning_rate(optimizer_1, epoch, epochs + 1, args.lr, cosine=True, lr_decay_rate=0.1)

            print('Warmup Net1')
            self.train_base_one_epoch(epoch, net_1, optimizer_1, warmup_loader)

            # Test.
            test_acc = self.test_single_net(epoch, net_1, test_loader)
            # Log.
            self.info_dict['test_acc'].append(test_acc)

            log_acc_file.write(f'epoch={epoch:>3d}, accuracy={test_acc:.4f},'
                               f" average accuracy(last 10 epochs)={get_average_value(self.info_dict['test_acc'], last_n=10):.4f}\n")  # log accuracy.
            log_acc_file.flush()

            # Save model.
            if epoch % 1 == 0:
                self.save_model(epoch,
                                path=os.path.join(self.checkpoint_dir, f'epoch-{epoch}_test-acc-{test_acc:.4f}.pth'))

        # Destructor.
        log_acc_file.close()

    def evaluation(self):
        args = self.args
        net_1 = self.net_1
        net_2 = self.net_2

        test_dataset = self.dataset_getter.get_test_dataset()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)
        test_acc = self.test(-1, net_1, net_2, test_loader)

        return test_acc

    @staticmethod
    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def evaluate_detection(noisy_or_not_predict, noisy_or_not_real):
    precision = np.sum(noisy_or_not_predict * noisy_or_not_real) / np.sum(noisy_or_not_predict)
    recall = np.sum(noisy_or_not_predict * noisy_or_not_real) / np.sum(noisy_or_not_real)
    fscore = 2.0 * precision * recall / (precision + recall)

    return precision, recall, fscore


class DivideMixClusterNMCETrainCifar(DivideMixTrainCifar):
    def __init__(self, net_1, net_2, optimizer_1, optimizer_2, dataset_getter, loss_controller, device, args):
        super().__init__(net_1, net_2, optimizer_1, optimizer_2, dataset_getter, loss_controller, device, args)

        self.cluster_num_2 = None
        self.cluster_num_1 = None
        self.loss_controller.register_gumble_softmax(args.tau)
        self.loss_controller.register_mcrr_loss(args.eps)

        # Constructor.
        self.clusters_1 = None
        self.clusters_2 = None

        # Path.
        self.cluster_dir = os.path.join(self.root_dir, 'cluster')
        self.cluster_info_dir = os.path.join(self.cluster_dir, 'info')
        self.img_cluster_dir = os.path.join(self.cluster_dir, 'cluster_imgs_ep')
        self.pca_figures_dir = os.path.join(self.cluster_dir, 'pca_figures')
        utils.makedirs(self.cluster_dir)
        utils.makedirs(self.cluster_info_dir)
        utils.makedirs(self.img_cluster_dir)
        utils.makedirs(self.pca_figures_dir)

    def save_cluster_info(self, output: str, epoch: int = -1) -> bool:
        file_path = os.path.join(self.cluster_info_dir, f'epoch-{epoch}.txt')
        with open(file_path, 'w') as f:
            f.write(output)

        return True

    @staticmethod
    def info_cluster(cluster_noise_rate, cluster_sample_num):
        index = np.argsort(cluster_noise_rate)

        result_str = ""
        for idx in index:
            nr = cluster_noise_rate[idx]
            sm = cluster_sample_num[idx]

            result_str += f"cluster {idx:>3d}| sample num: {sm:>4d}, noise rate: {nr:.4f}" + '\n'

        return result_str

    @staticmethod
    def chunk_avg(x, n_chunks=2, normalize=False):
        x_list = x.chunk(n_chunks, dim=0)
        x = torch.stack(x_list, dim=0)
        if not normalize:
            return x.mean(0)
        else:
            return F.normalize(x.mean(0), dim=1)

    def cluster_features(self, net_1, net_2, eval_loader, cluster_num, epoch=0, interval=1, cluster_method='kmeans'):

        if epoch % interval == 0 or self.clusters_1 is None or self.clusters_2 is None:
            features_1, targets_1 = self.feature_extractor(net_1, eval_loader)
            features_2, targets_2 = self.feature_extractor(net_2, eval_loader)
            if cluster_method == 'kmeans':
                clusters_1 = cluster_kmeans(features_1, cluster_num=cluster_num)
                clusters_2 = cluster_kmeans(features_2, cluster_num=cluster_num)
                cluster_num_1, cluster_num_2 = cluster_num, cluster_num
            elif cluster_method == 'kmeans++':
                clusters_1 = cluster_kmeans_plusplus(features_1, cluster_num=cluster_num)
                clusters_2 = cluster_kmeans_plusplus(features_2, cluster_num=cluster_num)
                cluster_num_1, cluster_num_2 = cluster_num, cluster_num
            elif cluster_method == 'spectral_clustering':
                clusters_1 = cluster_SpectralClustering(features_1, cluster_num=cluster_num)
                clusters_2 = cluster_SpectralClustering(features_2, cluster_num=cluster_num)
                cluster_num_1, cluster_num_2 = cluster_num, cluster_num
            elif cluster_method == 'agglomerative_clustering':
                clusters_1 = cluster_AgglomerativeClustering(features_1, cluster_num=cluster_num)
                clusters_2 = cluster_AgglomerativeClustering(features_2, cluster_num=cluster_num)
                cluster_num_1, cluster_num_2 = cluster_num, cluster_num
            elif cluster_method == 'dbscan':
                clusters_1 = cluster_dbscan(features_1)
                clusters_2 = cluster_dbscan(features_2)
                cluster_num_1 = len(set(clusters_1))
                cluster_num_2 = len(set(clusters_2))
            else:
                raise NotImplementedError
            self.clusters_1 = clusters_1
            self.clusters_2 = clusters_2
            self.cluster_num_1 = cluster_num_1
            self.cluster_num_2 = cluster_num_2

        else:
            clusters_1 = self.clusters_1
            clusters_2 = self.clusters_2
            cluster_num_1 = self.cluster_num_1
            cluster_num_2 = self.cluster_num_2

        return clusters_1, clusters_2, cluster_num_1, cluster_num_2

    def feature_extractor(self, net, data_loader):
        # Parameters.
        device = self.device

        net.eval()

        feature_arr = []
        target_arr = []
        with torch.no_grad():
            for iter_num, (data, target, index) in enumerate(data_loader):
                data = data.to(device)
                features = net(data, 'return_feature_only')
                features = torch.flatten(features, 1).cpu()

                feature_arr.append(features)
                target_arr.append(target)

        feature_arr = torch.cat(feature_arr)
        target_arr = torch.cat(target_arr)
        return feature_arr, target_arr

    def test_cluster_distribution_kernel(self, net, data_loader):
        # Parameters.
        device = self.device
        args = self.args

        net.eval()
        label_list = list()
        cluster_list = list()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                logit, z = net(inputs, 'return_clustering_and_subspace')
                if logit.sum() == 0:
                    logit += torch.randn_like(logit)

                cluster_list.append(logit.max(dim=1)[1].cpu())
                label_list.append(targets.cpu())

        cluster_list = torch.cat(cluster_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        clusters_info = torch.bincount(cluster_list, minlength=args.cluster_num)
        print(clusters_info, len(clusters_info), torch.nonzero(clusters_info).shape)

    def test_cluster_distribution(self):
        # Parameters.
        args = self.args
        net_1 = self.net_1
        net_2 = self.net_2

        test_dataset = self.dataset_getter.get_test_dataset()
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False,
                                 num_workers=8)
        self.test_cluster_distribution_kernel(net_1, test_loader)
        # self.test_cluster_distribution_kernel(net_2, test_loader)

    def eval_cluster_distribution(self):
        # Parameters.
        args = self.args
        net_1 = self.net_1
        net_2 = self.net_2

        eval_dataset = self.dataset_getter.get_warmup_dataset()
        eval_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False,
                                 num_workers=8)
        self.test_cluster_distribution_kernel(net_1, eval_loader)
        # self.test_cluster_distribution_kernel(net_2, test_loader)

    def eval_prob_clean_for_dataset_division_clusters(self, net, data_loader, clusters, cluster_num=10):
        # Parameters
        args = self.args
        device = self.device

        ce_loss_none = self.loss_controller.cross_entropy_none

        net.eval()
        if args.dataset == 'cifar10':
            losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 10))
        elif args.dataset == 'cifar100':
            losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 100))
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch_idx, (inputs, targets, index) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = ce_loss_none(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[index[b]] = loss[b]
        losses = (losses - losses.min()) / (losses.max() - losses.min()).cpu().numpy()

        # fit a two-component GMM to the loss
        input_loss = losses.reshape(-1, 1)

        cluster_elements_num_arr = list()

        prob = np.zeros(len(input_loss))
        for c in range(cluster_num):
            index = (clusters == c)
            loss_part = input_loss[index]

            cluster_elements_num_arr.append(len(loss_part))

            if len(loss_part) == 1:
                prob_part = 0
            else:
                gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
                gmm.fit(loss_part)

                prob_part = gmm.predict_proba(loss_part)
                prob_part = prob_part[:, gmm.means_.argmin()]

            prob[index] = prob_part

        print(sorted(cluster_elements_num_arr))
        return prob

    def eval_cluster_noise_rate(self, label_true, label_noisy, clusters, cluster_num=10):
        # Parameters.
        args = self.args

        cluster_noise_rate = np.zeros(cluster_num)
        cluster_sample_num = np.zeros(cluster_num, dtype=int)
        for c in range(cluster_num):
            index = (clusters == c)
            label_true_part = label_true[index]
            label_noisy_part = label_noisy[index]

            cluster_noise_rate[c] = np.mean(np.not_equal(label_true_part, label_noisy_part))
            cluster_sample_num[c] = len(label_true_part)

        return cluster_noise_rate, cluster_sample_num

    def eval_detection(self, net_1, net_2, data_loader, data_getter):
        # Parameters
        args = self.args
        device = self.device

        net_1.eval()
        net_2.eval()

        predict_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = net_1(inputs)
                outputs2 = net_2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                predict_list.append(predicted.cpu())

        predict_labels = torch.cat(predict_list).numpy()
        clean_labels, noisy_labels = data_getter.train_labels_true, data_getter.train_labels_noisy

        detect_res = ~np.equal(predict_labels, noisy_labels)
        noise_res = ~np.equal(noisy_labels, clean_labels)

        precision, recall, f1_score = evaluate_detection(detect_res, noise_res)
        print(f"===> Detection Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}\n")

    def test_cluster_acc(self, test_loader, print_result=False, save_name_img='cluster_img',
                         save_name_fig='pca_figure'):

        # Parameters.
        device = self.device
        net_1 = self.net_1
        net_2 = self.net_2

        cluster_list = []
        label_list = []
        x_list = []
        z_list = []

        net_1.eval()
        net_2.eval()
        for x, y in test_loader:
            with torch.no_grad():
                x, y = x.float().to(device), y.to(device)
                logit_1, z_1 = net_1(x, 'return_clustering_and_subspace')
                logit_2, z_2 = net_2(x, 'return_clustering_and_subspace')
                logit = (logit_1 + logit_2) / 2
                z = (z_1 + z_2) / 2

                if logit.sum() == 0:
                    logit += torch.randn_like(logit)
                cluster_list.append(logit.max(dim=1)[1].cpu())
                label_list.append(y.cpu())
                x_list.append(x.cpu())
                z_list.append(z.cpu())

        cluster_mtx = torch.cat(cluster_list, dim=0)
        label_mtx = torch.cat(label_list, dim=0)
        x_mtx = torch.cat(x_list, dim=0)
        z_mtx = torch.cat(z_list, dim=0)
        _, _, acc_single = cluster_match(cluster_mtx, label_mtx, n_classes=label_mtx.max() + 1, print_result=False)
        _, _, acc_merge = cluster_merge_match(cluster_mtx, label_mtx, print_result=False)
        NMI = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
        ARI = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
        if print_result:
            print('cluster match acc {}, cluster merge match acc {}, NMI {}, ARI {}'.format(acc_single, acc_merge, NMI,
                                                                                            ARI))

        save_name_img += '_acc' + str(acc_single)[2:5]
        save_name_img = os.path.join(self.cluster_dir, save_name_img)
        save_name_fig = os.path.join(self.cluster_dir, save_name_fig)

        save_cluster_imgs(cluster_mtx, x_mtx, save_name_img)
        save_latent_pca_figure(z_mtx, cluster_mtx, save_name_fig)

        return acc_single, acc_merge

    def train_nmce_one_epoch(self, epoch, net, optimizer, data_loader):
        # Parameters.
        args = self.args
        device = self.device
        gumble_softmax = self.loss_controller.gumble_softmax
        criterion = self.loss_controller.mcrr_loss

        net.train()

        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = torch.cat(inputs, dim=0)
            inputs = inputs.to(device)

            logits, z = net(inputs, 'return_clustering_and_subspace')
            prob = gumble_softmax(logits)
            if args.n_views > 1:
                z_avg = self.chunk_avg(z, n_chunks=args.n_views, normalize=True)
                prob = self.chunk_avg(prob, n_chunks=args.n_views)
            else:
                z_avg = z

            loss, loss_list = criterion(z_avg, prob, num_classes=args.cluster_num)
            z_list = z.chunk(args.n_views, dim=0)
            z_sim = (z_list[0] * z_list[1]).sum(1).mean()

            loss = loss - args.z_weight * z_sim
            loss_list += [z_sim.item()]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_tcr_one_epoch(self, epoch, net, optimizer, data_loader):
        # Parameters.
        args = self.args
        device = self.device

        criterion = TotalCodingRate(eps=args.eps)
        criterion_z = Z_loss()

        net.train()
        warmup_lr(optimizer, epoch, args.lr, warmup_epoch=10)
        with tqdm(total=len(data_loader)) as progress_bar:
            for step, (x, y) in enumerate(data_loader):
                x = torch.cat(x, dim=0)
                x, y = x.float().to(device), y.to(device)

                z = net(x, 'return_subspace_only')
                # calculate cosine similarity between z vectors for reference
                loss_z, z_sim = criterion_z(z)

                z_list = z.chunk(2, dim=0)
                loss = (criterion(z_list[0]) + criterion(z_list[1])) / 2
                loss_t = loss + args.z_weight * loss_z

                loss_list = [z_sim.item()]

                optimizer.zero_grad()
                loss_t.backward()
                optimizer.step()

                progress_bar.set_description(str(epoch))
                progress_bar.set_postfix(loss=loss.item(),
                                         z_sim=z_sim.item(),
                                         lr=optimizer.param_groups[0]['lr']
                                         )
                progress_bar.update(1)

    def train(self, epochs, epochs_warmup):
        # Parameters
        optimizer_1 = self.optimizer_1
        optimizer_2 = self.optimizer_2
        net_1 = self.net_1
        net_2 = self.net_2

        args = self.args
        lr = args.lr

        # Constructor.
        log_acc_file = open(os.path.join(self.root_dir, 'acc.log'), 'w')

        # Data loader.
        warmup_dataset = self.dataset_getter.get_warmup_dataset()
        warmup_loader = torch.utils.data.DataLoader(warmup_dataset, batch_size=2 * args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, drop_last=False)
        eval_dataset = self.dataset_getter.get_eval_dataset()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)
        test_dataset = self.dataset_getter.get_test_dataset()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)
        # nmce_dataset = self.dataset_getter.get_nmce_dataset(n_views=args.n_views)
        # nmce_loader = torch.utils.data.DataLoader(nmce_dataset, batch_size=args.batch_size, shuffle=True,
        #                                           num_workers=args.num_workers, drop_last=False)

        for epoch in range(epochs + 1):

            # learning rate schedule
            if epoch == epochs / 2:
                lr /= 10
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr
            # adjust_learning_rate(optimizer_1, epoch, epochs + 1, args.lr, cosine=True, lr_decay_rate=0.1)
            # adjust_learning_rate(optimizer_2, epoch, epochs + 1, args.lr, cosine=True, lr_decay_rate=0.1)

            # Train warmup.
            if epoch < epochs_warmup:
                print('Warmup Net1')
                self.warmup_one_epoch(epoch, net_1, optimizer_1, warmup_loader)
                print('\nWarmup Net2')
                self.warmup_one_epoch(epoch, net_2, optimizer_2, warmup_loader)
                print('\n' * 2)

            # Train dividemix.
            else:
                # Feature extraction and cluster.
                cluster_num = args.cluster_num
                clusters_1, clusters_2, cluster_num_1, cluster_num_2 = self.cluster_features(net_1, net_2, eval_loader,
                                                                                             cluster_num,
                                                                                             epoch=epoch - epochs_warmup,
                                                                                             interval=args.cluster_interval,
                                                                                             cluster_method=args.cluster_method)

                prob_1 = self.eval_prob_clean_for_dataset_division_clusters(net_1, eval_loader, clusters_1, cluster_num_1)
                prob_2 = self.eval_prob_clean_for_dataset_division_clusters(net_2, eval_loader, clusters_2, cluster_num_2)
                pred_1 = np.array(prob_1 > args.p_threshold, dtype=bool)
                pred_2 = np.array(prob_2 > args.p_threshold, dtype=bool)

                # for i in range(cluster_num):
                #     label_main = np.argmax(np.bincount(eval_dataset.label[clusters_1 == i]))
                #     pred_clean = pred_1[clusters_1 == i]
                #     labels_clean = eval_dataset.label[clusters_1 == i][pred_clean]
                #     labels_corrupt = eval_dataset.label[clusters_1 == i][~pred_clean]
                #     print(f'Cluster {i} has {len(pred_clean)} samples,'
                #           f' main label {label_main}')
                #     print(f'cluster: {np.bincount(eval_dataset.label[clusters_1 == i])}')
                #     print(f'Clean: {np.bincount(labels_clean)}')
                #     print(f'Corrupt: {np.bincount(labels_corrupt)}')

                print(f"Labeled data has a size of {pred_1.sum()}, {pred_2.sum()}")
                print(f"Unlabeled data has a size of {np.logical_not(pred_1).sum()}, {np.logical_not(pred_2).sum()}")

                print('Train Net1')
                labeled_dataset, unlabeled_dataset = self.dataset_getter.get_divide_datasets(pred=pred_2, prob=prob_2)
                labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_workers, drop_last=False)
                unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                               shuffle=True, num_workers=args.num_workers,
                                                               drop_last=False)

                self.train_semi_supervised_one_epoch(epoch, net_1, net_2, optimizer_1, labeled_loader, unlabeled_loader)

                print('\nTrain Net2')
                labeled_dataset, unlabeled_dataset = self.dataset_getter.get_divide_datasets(pred=pred_1, prob=prob_1)
                labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.num_workers, drop_last=False)
                unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                               shuffle=True, num_workers=args.num_workers,
                                                               drop_last=False)
                self.train_semi_supervised_one_epoch(epoch, net_2, net_1, optimizer_2, labeled_loader, unlabeled_loader)
                print('\n' * 2)

            # # Train NMCE.
            # if epoch >= epochs_warmup:
            #     # self.train_nmce_one_epoch(epoch, net_1, optimizer_1, nmce_loader)
            #     # self.train_nmce_one_epoch(epoch, net_2, optimizer_2, nmce_loader)
            #     self.train_tcr_one_epoch(epoch, net_1, optimizer_1, nmce_loader)
            #     self.train_tcr_one_epoch(epoch, net_2, optimizer_2, nmce_loader)

            # Test.
            test_acc = self.test(epoch, net_1, net_2, test_loader)
            # self.test_cluster_acc(test_loader, print_result=True)

            # Detection accuracy.
            if epoch % 10 == 0:
                self.eval_detection(net_1, net_2, eval_loader, self.dataset_getter)

            # Log.
            self.info_dict['test_acc'].append(test_acc)

            log_acc_file.write(f'epoch={epoch:>3d}, accuracy={test_acc:.4f},'
                               f" average accuracy(last 10 epochs)={get_average_value(self.info_dict['test_acc'], last_n=10):.4f}\n")  # log accuracy.
            log_acc_file.flush()

            if epoch >= epochs_warmup:
                cluster_noise_rate, cluster_sample_num = self.eval_cluster_noise_rate(
                    self.dataset_getter.train_labels_true,
                    self.dataset_getter.train_labels_noisy,
                    self.clusters_1,
                    cluster_num=cluster_num)
                self.save_cluster_info(self.info_cluster(cluster_noise_rate, cluster_sample_num), epoch)

            # Save model.
            # if epoch % 1 == 0:
            #     self.save_model(epoch,
            #                     path=os.path.join(self.checkpoint_dir, f'epoch-{epoch}_test-acc-{test_acc:.4f}.pth'))

        # Destructor.
        log_acc_file.close()

    def train_nmce_selfsup(self, epochs=10):

        import repositories.NMCE.NMCE.utils as utils

        args = self.args
        device = self.device
        use_fp16 = True

        model_dir = os.path.join(self.root_dir, 'nmce_draft', 'selfsup_{}'.format(args.net))

        utils.init_pipeline(model_dir)

        criterion = TotalCodingRate(eps=args.eps)
        criterion_z = Z_loss()

        net = self.net_1

        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = self.dataset_getter.get_nmce_dataset(n_views=args.n_views, transform=aug_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=16)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,
                              nesterov=False)
        optimizer = LARSWrapper(optimizer, eta=0.02, clip=True, exclude_bias_n_norm=True)

        scaler = GradScaler()
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)

        if use_fp16:
            print('using fp16 precision')

        ## Training
        for epoch in range(epochs):
            warmup_lr(optimizer, epoch, args.lr, warmup_epoch=10)
            with tqdm(total=len(train_loader)) as progress_bar:
                for step, (x, y) in enumerate(train_loader):
                    x = torch.cat(x, dim=0)
                    x, y = x.float().to(device), y.to(device)

                    with autocast(enabled=use_fp16):
                        z = net(x, 'return_subspace_only')
                        # calculate cosine similarity between z vectors for reference
                        loss_z, z_sim = criterion_z(z)

                        z_list = z.chunk(2, dim=0)
                        loss = (criterion(z_list[0]) + criterion(z_list[1])) / 2
                        loss_t = loss + args.z_weight * loss_z

                    loss_list = [z_sim.item()]

                    optimizer.zero_grad()
                    if use_fp16:
                        scaler.scale(loss_t).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_t.backward()
                        optimizer.step()

                    progress_bar.set_description(str(epoch))
                    progress_bar.set_postfix(loss=loss.item(),
                                             z_sim=z_sim.item(),
                                             lr=optimizer.param_groups[0]['lr']
                                             )
                    progress_bar.update(1)

            scheduler.step()
        print("training complete.")

    def train_nmce_clustering_kernel(self, net, train_loader, test_loader, epochs=10, train_backbone=False,
                                     use_fp16=False):

        args = self.args

        # GPU setup
        device = self.device
        ## get model directory
        model_dir = os.path.join(self.root_dir, 'nmce_draft', 'selfsup_{}'.format(self.args.net))

        # model
        G_Softmax = Gumble_Softmax(self.args.tau)

        # loss
        criterion = MaximalCodingRateReduction(eps=self.args.eps, gamma=1.0)

        # only optimize cluster and subspace module
        if train_backbone:
            para_list = [p for p in net.backbone.parameters()] + [p for p in net.subspace.parameters()]
            para_list_c = [p for p in net.cluster.parameters()]
        else:
            para_list = [p for p in net.subspace.parameters()]
            para_list_c = [p for p in net.cluster.parameters()]
            net.freeze_supervised()

        optimizer = optim.SGD(para_list, lr=self.args.lr, momentum=0.9, weight_decay=0.005, nesterov=False)
        optimizer = LARSWrapper(optimizer, eta=0.02, clip=True, exclude_bias_n_norm=True)

        optimizerc = optim.SGD(para_list_c, lr=0.5, momentum=0.9, weight_decay=0.005, nesterov=False)
        scaler = GradScaler()
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, last_epoch=-1)

        ## Training
        for epoch in range(epochs):
            with tqdm(total=len(train_loader)) as progress_bar:
                for step, (x, y) in enumerate(train_loader):
                    x = torch.cat(x, dim=0)
                    x, y = x.float().to(device), y.to(device)

                    with autocast(enabled=use_fp16):
                        logits, z = net(x, 'return_clustering_and_subspace')

                        prob = G_Softmax(logits)

                        if self.args.n_views > 1:
                            z_avg = self.chunk_avg(z, n_chunks=self.args.n_views, normalize=True)
                            prob = self.chunk_avg(prob, n_chunks=self.args.n_views)

                        loss, loss_list = criterion(z_avg, prob, num_classes=self.args.cluster_num)

                    z_list = z.chunk(self.args.n_views, dim=0)
                    z_sim = (z_list[0] * z_list[1]).sum(1).mean()

                    loss = loss - args.z_weight * z_sim

                    loss_list += [z_sim.item()]

                    optimizer.zero_grad()
                    optimizerc.zero_grad()
                    if use_fp16:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizerc.step()
                    else:
                        loss.backward()
                        optimizer.step()
                        optimizerc.step()

                    progress_bar.set_description(str(epoch))
                    progress_bar.set_postfix(loss=-loss_list[0] + loss_list[1],
                                             loss_d=loss_list[0],
                                             loss_c=loss_list[1],
                                             z_sim=z_sim.item()
                                             )
                    progress_bar.update(1)
            scheduler.step()

            save_name_img = model_dir + '/cluster_imgs/cluster_imgs_ep' + str(epoch + 1)
            save_name_fig = model_dir + '/pca_figures/z_space_pca' + str(epoch + 1)
            self.cluster_acc(test_loader, net, device, print_result=True, save_name_img=save_name_img,
                             save_name_fig=save_name_fig)

        print("training complete.")

    def train_nmce_clustering(self, epochs=10, train_backbone=False):
        # data
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = self.dataset_getter.get_nmce_dataset(n_views=self.args.n_views, transform=aug_transform)
        test_dataset = self.dataset_getter.get_test_dataset()
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False,
                                 num_workers=8)

        self.train_nmce_clustering_kernel(self.net_1, train_loader, test_loader, epochs=epochs,
                                          train_backbone=train_backbone)
        # self.train_nmce_clustering_kernel(self.net_2, train_loader, test_loader, epochs=epochs, train_backbone=train_backbone)

    def cluster_acc(self, test_loader, net, device, print_result=False, save_name_img='cluster_img',
                    save_name_fig='pca_figure'):
        cluster_list = []
        label_list = []
        x_list = []
        z_list = []
        net.eval()
        for x, y in test_loader:
            with torch.no_grad():
                x, y = x.float().to(device), y.to(device)
                logit, z = net(x, 'return_clustering_and_subspace')
                if logit.sum() == 0:
                    logit += torch.randn_like(logit)
                cluster_list.append(logit.max(dim=1)[1].cpu())
                label_list.append(y.cpu())
                x_list.append(x.cpu())
                z_list.append(z.cpu())
        net.train()
        cluster_mtx = torch.cat(cluster_list, dim=0)
        label_mtx = torch.cat(label_list, dim=0)
        x_mtx = torch.cat(x_list, dim=0)
        z_mtx = torch.cat(z_list, dim=0)
        _, _, acc_single = cluster_match(cluster_mtx, label_mtx, n_classes=label_mtx.max() + 1, print_result=False)
        _, _, acc_merge = cluster_merge_match(cluster_mtx, label_mtx, print_result=False)
        NMI = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
        ARI = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
        if print_result:
            print(
                'cluster match acc {}, cluster merge match acc {}, NMI {}, ARI {}'.format(acc_single, acc_merge, NMI,
                                                                                          ARI))

        # save_name_img += '_acc' + str(acc_single)[2:5]
        # save_cluster_imgs(cluster_mtx, x_mtx, save_name_img)
        # save_latent_pca_figure(z_mtx, cluster_mtx, save_name_fig)

        return acc_single, acc_merge


def adjust_learning_rate(optimizer, epoch, num_epochs, lr_init, cosine=True, lr_decay_rate=0.1):
    lr = lr_init
    if cosine:
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / num_epochs)) / 2
    else:
        # steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        # if steps > 0:
        #     lr = lr * (lr_decay_rate ** steps)
        raise NotImplementedError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
