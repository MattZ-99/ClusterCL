# -*- coding: utf-8 -*-
# @Time : 2022/10/17 16:48
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

from .TrainModuleRobust import *
from sklearn.mixture import GaussianMixture


class TrainNoisyDataDualNetDivideMix(TrainNoisyDataDualNetWithWarmup):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetDivideMix, self).__init__(net, optimizer, loss_controller, dataset_generator, device, args)

    def eval_prob_clean_for_dataset_division(self, net, data_loader):
        # Parameters
        args = self.args
        device = self.device

        ce_loss_none = self.loss_controller.get_cross_entropy_loss_none_reduction()

        net.eval()

        losses = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = ce_loss_none(outputs, targets)
                losses.append(loss)
        losses = torch.cat(losses)
        losses = ((losses - losses.min()) / (losses.max() - losses.min())).cpu().numpy()

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
        criterion = self.loss_controller.get_semi_loss_for_cluster_mix(lambda_u=args.lambda_u)

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

    def train_supervised_with_augments_one_epoch_for_one_net(self, epoch: int, data_loader, net, optimizer, *args, **kwargs):
        # Parameters.
        args = self.args
        device = self.device
        net.train()

        # loss functions.
        loss_fn = self.loss_controller.get_cross_entropy_loss()
        fmix_module = self.loss_controller.get_fmix_module()

        loss_val = utils.ValueStat()
        acc_val_top_1 = utils.ValueStat()
        acc_val_top_5 = utils.ValueStat()

        for batch_idx, ([data, data_strong], targets, _) in enumerate(data_loader):
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

        prob1 = self.eval_prob_clean_for_dataset_division(net1, eval_loader)
        prob2 = self.eval_prob_clean_for_dataset_division(net2, eval_loader)
        pred1 = np.array(prob1 > args.p_threshold, dtype=bool)
        pred2 = np.array(prob2 > args.p_threshold, dtype=bool)

        self.save_preds(pred1, pred2)

        print(f"Labeled data has a size of {pred1.sum()}, {pred2.sum()}")
        print(f"Unlabeled data has a size of {np.logical_not(pred1).sum()}, {np.logical_not(pred2).sum()}")

        # TODO: Train the two nets.
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

        return None

    def save_preds(self, pred1, pred2):
        pred = np.stack([pred1, pred2], axis=0)
        np.save(os.path.join(self.args.output_dir, 'pred.npy'), pred)


class TrainNoisyDataDualNetDivideMixSupervised(TrainNoisyDataDualNetDivideMix):
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

        prob1 = self.eval_prob_clean_for_dataset_division(net1, eval_loader)
        prob2 = self.eval_prob_clean_for_dataset_division(net2, eval_loader)
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
