# -*- coding: utf-8 -*-
# @Time : 2022/10/17 22:53
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

from .TrainModuleRobust import *
from sklearn.mixture import GaussianMixture
from .tools.cluster_funcs import *
from .tools.funcs import linear_ramp_up


class TrainNoisyDataDualNetProMix(TrainNoisyDataDualNetWithWarmup):
    def __init__(self, net, optimizer, loss_controller, dataset_generator, device, args):
        super(TrainNoisyDataDualNetProMix, self).__init__(net, optimizer, loss_controller, dataset_generator, device,
                                                          args)

    def eval_prob_clean_for_dataset_division(self, net, data_loader, rho=0.5):
        # Parameters
        args = self.args
        device = self.device

        ce_loss_none = self.loss_controller.get_cross_entropy_loss_none_reduction()

        net.eval()
        losses = []
        targets_list = []
        num_class = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                num_class = outputs.shape[1]
                loss = ce_loss_none(outputs, targets)

                losses.append(loss)
                targets_list.append(targets)

        losses = torch.cat(losses).cpu()
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        targets_list = torch.cat(targets_list).cpu()

        prob = np.zeros(targets_list.shape[0])
        idx_chosen_sm = []
        min_len = 1e10
        for j in range(num_class):
            indices = np.where(targets_list.cpu().numpy() == j)[0]
            # torch.where will cause device error
            if len(indices) == 0:
                continue
            bs_j = targets_list.shape[0] * (1. / num_class)
            pseudo_loss_vec_j = losses[indices]
            sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(bs_j * rho)), len(indices)), 1)
            # at least one example
            idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
            min_len = min(min_len, partition_j)

        idx_chosen_sm = np.concatenate(idx_chosen_sm)
        prob[idx_chosen_sm] = 1

        return prob

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

            # TODO: use pseudo_label_1 here?
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

        prob_1 = self.eval_prob_clean_for_dataset_division(net1, eval_loader)
        prob_2 = self.eval_prob_clean_for_dataset_division(net2, eval_loader)
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
