# -*- coding: utf-8 -*-
# @Time : 2022/10/17 20:00
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

import time

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .TrainModuleRobust import *
from .TrainModuleDivideMix import *
from sklearn.mixture import GaussianMixture
from .tools.cluster_funcs import *
from .tools.funcs import linear_ramp_up
from .tools.dimension_reduction_funcs import *


class TrainNoisyDataDualNetClusterMix(TrainNoisyDataDualNetDivideMix):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetClusterMix, self).__init__(net, optimizer, loss_controller, dataset_generator,
                                                              device, args)

        # Constructor.
        self.clusters_1 = None
        self.clusters_2 = None
        self.cluster_num_1 = None
        self.cluster_num_2 = None
        self.clustering_count = 0

    def feature_extractor(self, net, data_loader):
        # Parameters.
        device = self.device

        net.eval()

        feature_arr = []
        target_arr = []
        with torch.no_grad():
            for iter_num, (data, target) in enumerate(data_loader):
                data = data.to(device)
                features = net(data, 'return_features')
                features = torch.flatten(features, 1).cpu()

                feature_arr.append(features)
                target_arr.append(target)

        feature_arr = torch.cat(feature_arr)
        target_arr = torch.cat(target_arr)
        return feature_arr, target_arr

    def feature_dimension_reduction(self, feature_arr, dimension_reduction_method=None, dimension_reduction_num=100):
        if dimension_reduction_method is None:
            return feature_arr
        elif dimension_reduction_method == 'pca':
            return dimension_reduction_pca(feature_arr, dimension_reduction_num)
        elif dimension_reduction_method == 'ipca':
            return dimension_reduction_ipca(feature_arr, dimension_reduction_num)
        elif dimension_reduction_method == 'tsne':
            return dimension_reduction_tsne(feature_arr, dimension_reduction_num)
        elif dimension_reduction_method == 'umap':
            return dimension_reduction_umap(feature_arr, dimension_reduction_num)
        else:
            raise NotImplementedError

    def cluster_features(self, net_1, net_2, eval_loader, cluster_num, epoch=0, interval=1, cluster_method='kmeans',
                         dimension_reduction_method=None, dimension_reduction_num=100):

        if self.clustering_count % interval == 0 or self.clusters_1 is None or self.clusters_2 is None:
            print(f"Clustering at epoch {epoch}")
            time_cluster_start = time.time()

            features_1, targets_1 = self.feature_extractor(net_1, eval_loader)
            features_2, targets_2 = self.feature_extractor(net_2, eval_loader)

            # Feature Reduction.
            dimension_reduction_num = min(dimension_reduction_num,
                                          features_1.shape[1]) if dimension_reduction_method is not None else \
                features_1.shape[1]
            print(f"Perform ({dimension_reduction_method}) dimension reduction."
                  f" Feature dimension: {features_1.shape[1]} -> {dimension_reduction_num}.")
            features_1 = self.feature_dimension_reduction(features_1, dimension_reduction_method,
                                                          dimension_reduction_num)
            features_2 = self.feature_dimension_reduction(features_2, dimension_reduction_method,
                                                          dimension_reduction_num)

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

            print(f"Clustering finished. Time cost: {time.time() - time_cluster_start:.2f}s.")

            rtn_dict = {
                "clusters_1": clusters_1,
                "clusters_2": clusters_2,
                "cluster_num_1": cluster_num_1,
                "cluster_num_2": cluster_num_2,
                "features_1": features_1,
                "features_2": features_2,
            }

        else:
            clusters_1 = self.clusters_1
            clusters_2 = self.clusters_2
            cluster_num_1 = self.cluster_num_1
            cluster_num_2 = self.cluster_num_2

            rtn_dict = {}

        self.clustering_count += 1

        return clusters_1, clusters_2, cluster_num_1, cluster_num_2, rtn_dict

    def eval_prob_clean_for_dataset_division_clusters(self, net, data_loader, clusters, cluster_num=10):
        # Parameters
        args = self.args
        device = self.device

        ce_loss_none = self.loss_controller.get_cross_entropy_loss_none_reduction()

        net.eval()
        # if args.dataset == 'cifar10':
        #     losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 10))
        # elif args.dataset == 'cifar100':
        #     losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 100))
        # else:
        #     raise NotImplementedError
        losses = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = ce_loss_none(outputs, targets)
                losses.append(loss)
        losses = torch.cat(losses).cpu()
        losses = (losses - losses.min()) / (losses.max() - losses.min()).numpy()

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

        # print(sorted(cluster_elements_num_arr))
        return prob, input_loss

    def train_main_one_epoch(self, epoch: int, *args, **kwargs):
        # Parameters.
        args = self.args

        # Nets.
        net1 = self.net.net1
        net2 = self.net.net2

        # Data loader.
        eval_dataset = self.dataset_generator.generate_eval_noisy_dataset()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        # Feature extraction and cluster.
        cluster_num = args.cluster_num
        clusters_1, clusters_2, cluster_num_1, cluster_num_2, rtn_dict = self.cluster_features(
            net1, net2, eval_loader, cluster_num, epoch=epoch,
            interval=args.cluster_interval, cluster_method=args.cluster_method,
            dimension_reduction_method=args.dimension_reduction_method,
            dimension_reduction_num=args.dimension_reduction_components
        )

        prob1, loss_1 = self.eval_prob_clean_for_dataset_division_clusters(net1, eval_loader, clusters_1, cluster_num_1)
        prob2, loss_2 = self.eval_prob_clean_for_dataset_division_clusters(net2, eval_loader, clusters_2, cluster_num_2)

        pred1 = np.array(prob1 > args.p_threshold, dtype=bool)
        pred2 = np.array(prob2 > args.p_threshold, dtype=bool)

        self.save_preds(pred1, pred2)

        utils.save_pickle(
            {
                "fc": rtn_dict,
                "loss_1": loss_1,
                "loss_2": loss_2
            },
            os.path.join(self.args.output_dir, f'fc-{epoch}.pickle')
        )

        print(f"Labeled data has a size of {pred1.sum()}, {pred2.sum()}")
        print(f"Unlabeled data has a size of {np.logical_not(pred1).sum()}, {np.logical_not(pred2).sum()}")

        print('Train Net1')
        labeled_dataset, unlabeled_dataset = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob2, pred=pred2)
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=False)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.num_workers,
                                                       drop_last=False)

        self.train_semi_supervised_one_epoch(epoch, net1, net2, self.optimizer[0], labeled_loader, unlabeled_loader)
        # self.train_basic_one_epoch_for_one_net(epoch, labeled_loader, net1, self.optimizer[0])

        print('Train Net2')
        labeled_dataset, unlabeled_dataset = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob1, pred=pred1)
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=False)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.num_workers,
                                                       drop_last=False)
        self.train_semi_supervised_one_epoch(epoch, net2, net1, self.optimizer[1], labeled_loader, unlabeled_loader)
        # self.train_basic_one_epoch_for_one_net(epoch, labeled_loader, net2, self.optimizer[1])

        # calculate precision, recall.
        train_label_clean = self.dataset_generator.train_label_clean_or_not
        precision, recall, f1 = funcs.calculate_precision_recall_f1(pred1, train_label_clean)
        print(f"Network1-Precision: {precision}, Recall: {recall}, F1: {f1}")
        precision, recall, f1 = funcs.calculate_precision_recall_f1(pred2, train_label_clean)
        print(f"Network2-Precision: {precision}, Recall: {recall}, F1: {f1}")

        return None


class TrainNoisyDataDualNetClusterMixPlus(TrainNoisyDataDualNetWithWarmup):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetClusterMixPlus, self).__init__(net, optimizer, loss_controller, dataset_generator,
                                                                  device, args)

        # Constructor.
        self.clusters_1 = None
        self.clusters_2 = None
        self.cluster_num_2 = None
        self.cluster_num_1 = None
        self.clustering_count = 0

    def feature_extractor(self, net, data_loader):
        # Parameters.
        device = self.device

        net.eval()

        feature_arr = []
        target_arr = []
        with torch.no_grad():
            for iter_num, (data, target) in enumerate(data_loader):
                data = data.to(device)
                features = net(data, 'return_features')
                features = torch.flatten(features, 1).cpu()

                feature_arr.append(features)
                target_arr.append(target)

        feature_arr = torch.cat(feature_arr)
        target_arr = torch.cat(target_arr)
        return feature_arr, target_arr

    def cluster_features(self, net_1, net_2, eval_loader, cluster_num, epoch=0, interval=1, cluster_method='kmeans'):

        if self.clustering_count % interval == 0 or self.clusters_1 is None or self.clusters_2 is None:
            print(f"Clustering at epoch {epoch}")
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

            self.clustering_count += 1

        else:
            clusters_1 = self.clusters_1
            clusters_2 = self.clusters_2
            cluster_num_1 = self.cluster_num_1
            cluster_num_2 = self.cluster_num_2

        return clusters_1, clusters_2, cluster_num_1, cluster_num_2

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

    def eval_prob_clean_for_dataset_division_clusters(self, net, data_loader, clusters, cluster_num=10):
        # Parameters
        args = self.args
        device = self.device

        ce_loss_none = self.loss_controller.get_cross_entropy_loss_none_reduction()

        net.eval()
        # if args.dataset == 'cifar10':
        #     losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 10))
        # elif args.dataset == 'cifar100':
        #     losses = torch.zeros(int(50000 * args.train_data_percent * args.num_class / 100))
        # else:
        #     raise NotImplementedError
        losses = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = ce_loss_none(outputs, targets)
                losses.append(loss)
        losses = torch.cat(losses).cpu()
        losses = (losses - losses.min()) / (losses.max() - losses.min()).numpy()

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

        # print(sorted(cluster_elements_num_arr))
        return prob

    @staticmethod
    def high_conf_sel2(idx_chosen, w_x, batch_size, score1, score2, match, tau=0.99, threshold=0.9):
        w_x2 = w_x.clone()
        if (1. * idx_chosen.shape[0] / batch_size) < threshold:
            # when clean data is insufficient, try to incorporate more examples
            high_conf_cond2 = (score1 > tau) * (score2 > tau) * match
            # both nets agrees
            high_conf_cond2 = (1. * high_conf_cond2 - w_x.squeeze()) > 0
            # remove already selected examples; newly selected
            hc2_idx = torch.where(high_conf_cond2)[0]

            max_to_sel_num = int(batch_size * threshold) - idx_chosen.shape[0]
            # maximally select batch_size * args.threshold; idx_chosen.shape[0] select already
            if high_conf_cond2.sum() > max_to_sel_num:
                # to many examples selected, remove some low conf examples
                score_mean = (score1 + score2) / 2
                idx_remove = (-score_mean[hc2_idx]).sort()[1][max_to_sel_num:]
                # take top scores
                high_conf_cond2[hc2_idx[idx_remove]] = False
            w_x2[high_conf_cond2] = 1
        return w_x2

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

    def train_one_epoch_semi_supervised(self, epoch, net1, net2, optimizer1, optimizer2, data_loader):
        net1.train()
        net2.train()

        device = self.device
        args = self.args
        loss_calculator = self.loss_ce_mixup_cr_for_one_net

        weight_for_loss = linear_ramp_up(epoch, 50 + args.epochs_warmup)
        num_iter = (len(data_loader.dataset) // args.batch_size) + 1

        # [[loss_ce, acc_top_1, acc_top_5] for net1, [loss_ce, acc_top_1, acc_top_5] for net2]
        vals_stat = utils.ValueListStat(shape=(2, 3))

        for batch_idx, ([data, data_strong], targets, w_x1, w_x2) in enumerate(data_loader):
            batch_size = data.size(0)

            # Transform label to one-hot vector
            targets = F.one_hot(targets, num_classes=args.num_class).float()
            w_x1 = w_x1.view(-1, 1).type(torch.FloatTensor)
            w_x2 = w_x2.view(-1, 1).type(torch.FloatTensor)

            data, data_strong, targets, w_x1, w_x2 = data.to(device), data_strong.to(device), targets.to(
                device), w_x1.to(device), w_x2.to(device)

            # label guessing.
            with torch.no_grad():
                outputs1 = net1(data)
                outputs2 = net2(data)

                px1 = torch.softmax(outputs1, dim=1)
                px2 = torch.softmax(outputs2, dim=1)
                pred_net1 = F.one_hot(px1.max(dim=1)[1], args.num_class).float()
                pred_net2 = F.one_hot(px2.max(dim=1)[1], args.num_class).float()

                high_conf_cond = (targets * px1).sum(dim=1) > args.tau
                high_conf_cond2 = (targets * px2).sum(dim=1) > args.tau
                w_x1[high_conf_cond] = 1
                w_x2[high_conf_cond2] = 1
                pseudo_label_1 = targets * w_x1 + pred_net1 * (1 - w_x1)
                pseudo_label_2 = targets * w_x2 + pred_net2 * (1 - w_x2)

                # selected examples
                idx_chosen_1 = torch.where(w_x1 == 1)[0]
                idx_chosen_2 = torch.where(w_x2 == 1)[0]

                if epoch > args.epochs - args.start_expand:
                    # only add these points at the last 100 epochs
                    score1 = px1.max(dim=1)[0]
                    score2 = px2.max(dim=1)[0]
                    match = px1.max(dim=1)[1] == px2.max(dim=1)[1]

                    assert isinstance(idx_chosen_1, torch.Tensor)
                    hc2_sel_wx1 = self.high_conf_sel2(idx_chosen_1, w_x1, batch_size, score1, score2, match)
                    hc2_sel_wx2 = self.high_conf_sel2(idx_chosen_2, w_x2, batch_size, score1, score2, match)

                    idx_chosen_1 = torch.where(hc2_sel_wx1 == 1)[0]
                    idx_chosen_2 = torch.where(hc2_sel_wx2 == 1)[0]
                # Label Guessing

            _, (loss_ce_1, loss_mixup_1, loss_fmix_1, loss_cr_1) = loss_calculator(net1, data[idx_chosen_1],
                                                                                   data_strong[idx_chosen_1],
                                                                                   targets[idx_chosen_1])
            # Entire loss.
            loss_1 = loss_ce_1 + weight_for_loss * (loss_mixup_1 + loss_fmix_1 + loss_cr_1)
            # Backward.
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()

            _, (loss_ce_2, loss_mixup_2, loss_fmix_2, loss_cr_2) = loss_calculator(net2, data[idx_chosen_2],
                                                                                   data_strong[idx_chosen_2],
                                                                                   targets[idx_chosen_2])
            # Entire loss.
            loss_2 = loss_ce_2 + weight_for_loss * (loss_mixup_2 + loss_fmix_2 + loss_cr_2)

            # Backward.
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()

            # target to class indices.
            targets = torch.argmax(targets, dim=1)

            # Log.
            acc_1_net1, acc_5_net1 = metrics.accuracy(outputs1, targets, topk=(1, 5))
            acc_1_net2, acc_5_net2 = metrics.accuracy(outputs2, targets, topk=(1, 5))
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

    def train_main_one_epoch(self, epoch: int, *args, **kwargs):
        # Parameters.
        args = self.args

        # Nets.
        net1 = self.net.net1
        net2 = self.net.net2

        # Data loader.
        eval_dataset = self.dataset_generator.generate_eval_noisy_dataset()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        # Feature extraction and cluster.
        cluster_num = args.cluster_num
        clusters_1, clusters_2, cluster_num_1, cluster_num_2 = self.cluster_features(net1, net2, eval_loader,
                                                                                     cluster_num,
                                                                                     epoch=epoch,
                                                                                     interval=args.cluster_interval,
                                                                                     cluster_method=args.cluster_method)

        prob_1 = self.eval_prob_clean_for_dataset_division_clusters(net1, eval_loader, clusters_1, cluster_num_1)
        prob_2 = self.eval_prob_clean_for_dataset_division_clusters(net2, eval_loader, clusters_2, cluster_num_2)
        pred_1 = np.array(prob_1 > args.p_threshold, dtype=bool)
        pred_2 = np.array(prob_2 > args.p_threshold, dtype=bool)

        print(f"Labeled data has a size of {pred_1.sum()}, {pred_2.sum()}")
        print(f"Unlabeled data has a size of {np.logical_not(pred_1).sum()}, {np.logical_not(pred_2).sum()}")

        dataset_with_preds = self.dataset_generator.generate_train_noisy_weak_strong_dataset_with_preds(pred_1, pred_2)
        data_loader_with_preds = torch.utils.data.DataLoader(dataset_with_preds, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=args.num_workers)
        result_dict = self.train_one_epoch_semi_supervised(epoch, net1, net2, self.optimizer[0], self.optimizer[1],
                                                           data_loader_with_preds)

        return result_dict


# TODO
class TrainNoisyDataDualNetClusterMixUpdateVersion1(TrainNoisyDataDualNetClusterMix):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super().__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def perform_label_co_refinement_for_labeled_data(self, net, data_loader):
        # Parameters
        args = self.args
        device = self.device

        label_co_refinement = []

        for batch_idx, ((inputs_x, inputs_x2), labels_x, w_x) in enumerate(data_loader):
            # Transform label to one-hot
            labels_x = F.one_hot(labels_x, num_classes=args.num_class)
            w_x = w_x.view(-1, 1).float()

            inputs_x, inputs_x2 = inputs_x.to(device), inputs_x2.to(device)
            labels_x, w_x = labels_x.to(device), w_x.to(device)

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temperature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            label_co_refinement.append(targets_x)

        return torch.cat(label_co_refinement, dim=0).cpu().numpy()

    def perform_label_co_guessing_for_unlabeled_data(self, net_1, net_2, data_loader):
        # Parameters
        args = self.args
        device = self.device

        label_co_guessing = []

        for batch_idx, (inputs_u, inputs_u2) in enumerate(data_loader):
            inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

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

            label_co_guessing.append(targets_u)

        return torch.cat(label_co_guessing, dim=0).cpu().numpy()

    def generate_pseudo_labels_for_one_net(self, net_1, net_2, labeled_loader, unlabeled_loader):
        # Parameters
        args = self.args
        device = self.device

        net_1.eval()
        net_2.eval()

        with torch.no_grad():
            label_co_refinement = self.perform_label_co_refinement_for_labeled_data(net_1, labeled_loader)
            label_co_guessing = self.perform_label_co_guessing_for_unlabeled_data(net_1, net_2, unlabeled_loader)

        return label_co_refinement, label_co_guessing

    def train_semi_supervised_one_epoch_update_version_1(self, epoch, net_1, net_2, optimizer, data_loader):
        # Parameters
        args = self.args
        device = self.device
        criterion = self.loss_controller.get_semi_loss_for_cluster_mix(lambda_u=args.lambda_u)

        # fix one network and train the other
        net_1.train()
        net_2.eval()
        num_iter = (len(data_loader.dataset) // args.batch_size) + 1
        for batch_idx, ((inputs_x,), labels_x, w_x) in enumerate(data_loader):
            inputs_x = inputs_x.to(device)
            labels_x, w_x = labels_x.to(device), w_x.to(device)

            # mixmatch
            lamb_mix = np.random.beta(args.alpha, args.alpha)
            lamb_mix = max(lamb_mix, 1 - lamb_mix)

            idx = torch.randperm(inputs_x.size(0))
            input_a, input_b = inputs_x, inputs_x[idx]
            target_a, target_b = labels_x, labels_x[idx]

            mixed_input = lamb_mix * input_a + (1 - lamb_mix) * input_b
            mixed_target = lamb_mix * target_a + (1 - lamb_mix) * target_b

            logits = net_1(mixed_input)
            logits_x, logits_u = logits[w_x], logits[~w_x]
            Lx, Lu, lamb = criterion(logits_x, mixed_target[w_x], logits_u, mixed_target[~w_x],
                                     epoch + batch_idx / num_iter, args.epochs_warmup)

            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.to(device)
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss_all = Lx + lamb * Lu + penalty

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

    def train_main_one_epoch_for_one_net_kernel(self, epoch, net1, net2, prob, pred):
        # Parameters
        args = self.args
        device = self.device

        # Label processing.
        labeled_dataset, unlabeled_dataset = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob, pred=pred)
        labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, drop_last=False)
        label_co_refinement, label_co_guessing = self.generate_pseudo_labels_for_one_net(net1, net2, labeled_loader,
                                                                                         unlabeled_loader)

        train_dataset = self.dataset_generator.generate_merged_labeled_and_unlabeled_dataset(
            label_co_refinement, label_co_guessing, pred)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=False)

        self.train_semi_supervised_one_epoch_update_version_1(epoch, net1, net2, self.optimizer[0], train_loader)

    def train_main_one_epoch(self, epoch: int, *args, **kwargs):
        # Parameters.
        args = self.args

        # Nets.
        net1 = self.net.net1
        net2 = self.net.net2

        # Data loader.
        eval_dataset = self.dataset_generator.generate_eval_noisy_dataset()
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False)

        # Feature extraction and cluster.
        cluster_num = args.cluster_num
        clusters_1, clusters_2, cluster_num_1, cluster_num_2 = self.cluster_features(
            net1, net2, eval_loader, cluster_num, epoch=epoch,
            interval=args.cluster_interval, cluster_method=args.cluster_method,
            dimension_reduction_method=args.dimension_reduction_method,
            dimension_reduction_num=args.dimension_reduction_components
        )

        prob1 = self.eval_prob_clean_for_dataset_division_clusters(net1, eval_loader, clusters_1, cluster_num_1)
        prob2 = self.eval_prob_clean_for_dataset_division_clusters(net2, eval_loader, clusters_2, cluster_num_2)

        pred1 = np.array(prob1 > args.p_threshold, dtype=bool)
        pred2 = np.array(prob2 > args.p_threshold, dtype=bool)

        print(f"Labeled data has a size of {pred1.sum()}, {pred2.sum()}")
        print(f"Unlabeled data has a size of {np.logical_not(pred1).sum()}, {np.logical_not(pred2).sum()}")

        print('Train Net1')
        self.train_main_one_epoch_for_one_net_kernel(epoch, net1, net2, prob2, pred2)

        print('Train Net2')
        self.train_main_one_epoch_for_one_net_kernel(epoch, net2, net1, prob1, pred1)

        return None


class TrainNoisyDataDualNetClusterMixPlusVersion2(TrainNoisyDataDualNetClusterMix):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super().__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    # Update: Add Fmix.
    def train_semi_supervised_one_epoch(self, epoch, net_1, net_2, optimizer, labeled_loader, unlabeled_loader):
        # Parameters
        args = self.args
        device = self.device
        criterion = self.loss_controller.get_semi_loss_for_cluster_mix(lambda_u=args.lambda_u)
        fmix_module = self.loss_controller.get_fmix_module()

        # fix one network and train the other
        net_1.train()
        net_2.eval()

        unlabeled_train_iter = iter(unlabeled_loader)
        num_iter = (len(labeled_loader.dataset) // args.batch_size) + 1

        for batch_idx, ((inputs_x, inputs_x2), labels_x, w_x) in enumerate(labeled_loader):
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

            # --------------------------------------------------------------------------------------------
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

            # --------------------------------------------------------------------------------------------
            # Fmix.
            fmix_input = fmix_module(all_inputs)
            outputs_fmix = net_1(fmix_input)
            loss_fmix = fmix_module.loss(outputs_fmix, all_targets)
            # --------------------------------------------------------------------------------------------
            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.to(device)
            pred_mean = torch.softmax(losses, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            # --------------------------------------------------------------------------------------------
            loss_all = Lx + lamb * Lu + penalty + loss_fmix

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()


class TrainNoisyDataDualNetClusterMixSupervised(TrainNoisyDataDualNetClusterMix):
    def train_main_one_epoch(self, epoch: int, *args, **kwargs):
        # Parameters.
        args = self.args

        # Nets.
        net1 = self.net.net1
        net2 = self.net.net2

        # Data loader.
        eval_dataset = self.dataset_generator.generate_eval_noisy_dataset()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        # Feature extraction and cluster.
        cluster_num = args.cluster_num
        clusters_1, clusters_2, cluster_num_1, cluster_num_2 = self.cluster_features(
            net1, net2, eval_loader, cluster_num, epoch=epoch,
            interval=args.cluster_interval, cluster_method=args.cluster_method,
            dimension_reduction_method=args.dimension_reduction_method,
            dimension_reduction_num=args.dimension_reduction_components
        )

        prob1 = self.eval_prob_clean_for_dataset_division_clusters(net1, eval_loader, clusters_1, cluster_num_1)
        prob2 = self.eval_prob_clean_for_dataset_division_clusters(net2, eval_loader, clusters_2, cluster_num_2)

        pred1 = np.array(prob1 > args.p_threshold, dtype=bool)
        pred2 = np.array(prob2 > args.p_threshold, dtype=bool)

        self.save_preds(pred1, pred2)

        print(f"Labeled data has a size of {pred1.sum()}, {pred2.sum()}")
        print(f"Unlabeled data has a size of {np.logical_not(pred1).sum()}, {np.logical_not(pred2).sum()}")

        # TODO: Train the two nets.
        print('Train Net1')
        labeled_dataset, _ = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob2, pred=pred2)
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=False)
        self.train_supervised_with_augments_one_epoch_for_one_net(epoch, labeled_loader, net1, self.optimizer[0])

        print('Train Net2')
        labeled_dataset, _ = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob1, pred=pred1)
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=False)
        self.train_supervised_with_augments_one_epoch_for_one_net(epoch, labeled_loader, net2, self.optimizer[1])

        return None


class TrainNoisyDataDualNetClusterMixWithContrastiveLoss(TrainNoisyDataDualNetClusterMix):
    # Parameters
    CL_size_C = 8

    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetClusterMixWithContrastiveLoss, self).__init__(net, optimizer, loss_controller,
                                                                                 dataset_generator,
                                                                                 device, args)

    def info_nce_loss(self, features):
        args = self.args
        device = self.device

        labels = torch.cat([torch.arange(args.batch_size * self.CL_size_C) for i in range(args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / args.T
        return logits, labels

    def train_semi_supervised_one_epoch_with_CL(self, epoch, net_1, net_2, optimizer, labeled_loader, unlabeled_loader,
                                                simclr_loader):
        # TODO: ADD contrastive loss.

        # Parameters
        args = self.args
        device = self.device
        criterion = self.loss_controller.get_semi_loss_for_cluster_mix(lambda_u=args.lambda_u)

        # fix one network and train the other
        net_1.train()
        net_2.eval()

        unlabeled_train_iter = iter(unlabeled_loader)
        num_iter = (len(labeled_loader.dataset) // args.batch_size) + 1

        simclr_train_iter = iter(simclr_loader)
        # num_iter_simclr = (len(simclr_loader.dataset) // args.batch_size) + 1

        for batch_idx, ((inputs_x, inputs_x2), labels_x, w_x) in enumerate(labeled_loader):
            try:
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_loader)
                inputs_u, inputs_u2 = next(unlabeled_train_iter)

            try:
                images_simclr = next(simclr_train_iter)
            except StopIteration:
                simclr_train_iter = iter(simclr_loader)
                images_simclr = next(simclr_train_iter)
            images_simclr = torch.cat(images_simclr, dim=0).to(device)

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

            # contrastive loss

            if epoch < 150:
                with autocast(enabled=False):
                    features = net_1(images_simclr, "return_features")

                    logits, labels = self.info_nce_loss(features)
                    loss_cl = F.cross_entropy(logits, labels)
            else:
                loss_cl = 0

            lamb_cl = 0.01 * (150 - epoch) / 150 + 1e-5 if epoch < 150 else 0

            loss_all = Lx + lamb * Lu + penalty + lamb_cl * loss_cl

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

    def train_main_one_epoch(self, epoch: int, *args, **kwargs):
        # Parameters.
        args = self.args

        # Nets.
        net1 = self.net.net1
        net2 = self.net.net2

        # Data loader.
        eval_dataset = self.dataset_generator.generate_eval_noisy_dataset()
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, drop_last=False)

        # SimCLR loader.
        simclr_dataset = self.dataset_generator.generate_train_simclr_dataset(n_views=args.n_views)
        simclr_loader = torch.utils.data.DataLoader(simclr_dataset, batch_size=args.batch_size * self.CL_size_C,
                                                    shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

        # Feature extraction and cluster.
        cluster_num = args.cluster_num
        clusters_1, clusters_2, cluster_num_1, cluster_num_2 = self.cluster_features(
            net1, net2, eval_loader, cluster_num, epoch=epoch,
            interval=args.cluster_interval, cluster_method=args.cluster_method,
            dimension_reduction_method=args.dimension_reduction_method,
            dimension_reduction_num=args.dimension_reduction_components
        )

        prob1 = self.eval_prob_clean_for_dataset_division_clusters(net1, eval_loader, clusters_1, cluster_num_1)
        prob2 = self.eval_prob_clean_for_dataset_division_clusters(net2, eval_loader, clusters_2, cluster_num_2)

        pred1 = np.array(prob1 > args.p_threshold, dtype=bool)
        pred2 = np.array(prob2 > args.p_threshold, dtype=bool)

        self.save_preds(pred1, pred2)

        print(f"Labeled data has a size of {pred1.sum()}, {pred2.sum()}")
        print(f"Unlabeled data has a size of {np.logical_not(pred1).sum()}, {np.logical_not(pred2).sum()}")

        print('Train Net1')
        labeled_dataset, unlabeled_dataset = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob2, pred=pred2)
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=False)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.num_workers,
                                                       drop_last=False)

        self.train_semi_supervised_one_epoch_with_CL(epoch, net1, net2, self.optimizer[0], labeled_loader,
                                                     unlabeled_loader, simclr_loader)

        print('Train Net2')
        labeled_dataset, unlabeled_dataset = self.dataset_generator.generate_train_noisy_divided_datasets(
            prob=prob1, pred=pred1)
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.num_workers, drop_last=False)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=args.num_workers,
                                                       drop_last=False)
        # self.train_semi_supervised_one_epoch_with_CL(epoch, net2, net1, self.optimizer[1], labeled_loader,
        #                                              unlabeled_loader, simclr_loader)
        self.train_semi_supervised_one_epoch(epoch, net2, net1, self.optimizer[1], labeled_loader, unlabeled_loader)

        # calculate precision, recall.
        train_label_clean = self.dataset_generator.train_label_clean_or_not
        precision, recall, f1 = funcs.calculate_precision_recall_f1(pred1, train_label_clean)
        print(f"Network1-Precision: {precision}, Recall: {recall}, F1: {f1}")
        precision, recall, f1 = funcs.calculate_precision_recall_f1(pred2, train_label_clean)
        print(f"Network2-Precision: {precision}, Recall: {recall}, F1: {f1}")

        return None
