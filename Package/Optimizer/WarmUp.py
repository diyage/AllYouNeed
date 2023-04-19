import torch
import math
import numpy as np


class WarmUpOptimizer:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            base_lr: float = 1e-3,
            warm_up_epoch: int = 1,
    ):
        self.optimizer = optimizer
        self.set_lr(base_lr)

        self.warm_up_epoch = warm_up_epoch
        self.base_lr = base_lr
        self.tmp_lr = base_lr

    def set_lr(self, lr):
        self.tmp_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warm(self,
             now_epoch_ind,
             now_batch_ind,
             max_batch_ind
             ):
        if now_epoch_ind < self.warm_up_epoch:
            self.tmp_lr = self.base_lr * pow(
                (now_batch_ind + now_epoch_ind * max_batch_ind) * 1. / (self.warm_up_epoch * max_batch_ind), 4)
            self.set_lr(self.tmp_lr)

        elif now_epoch_ind == self.warm_up_epoch and now_batch_ind == 0:
            self.tmp_lr = self.base_lr
            self.set_lr(self.tmp_lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


class WarmUpCosineAnnealOptimizer(WarmUpOptimizer):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_epoch_for_train: int,
            base_lr: float = 1e-3,
            warm_up_end_epoch: int = 1,
    ):
        super().__init__(
            optimizer,
            base_lr,
            warm_up_end_epoch
        )
        self.max_epoch_for_train = max_epoch_for_train

    def warm(
            self,
            now_epoch_ind,
            now_batch_ind,
            max_batch_ind
    ):
        if now_epoch_ind < self.warm_up_epoch:
            self.tmp_lr = self.base_lr * pow(
                (now_batch_ind + now_epoch_ind * max_batch_ind) * 1. / (self.warm_up_epoch * max_batch_ind), 4)
            self.set_lr(self.tmp_lr)
        else:
            T = (self.max_epoch_for_train - self.warm_up_epoch + 1) * max_batch_ind
            t = (now_epoch_ind - self.warm_up_epoch) * max_batch_ind + now_batch_ind

            lr = 1.0 / 2 * (1.0 + math.cos(t * math.pi / T)) * self.base_lr
            self.set_lr(lr)


class WarmUpAbsSineCircleOptimizer(WarmUpOptimizer):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            base_lr: float = 1e-3,
            reach_base_lr_cost_epoch: int = 1
    ):
        super().__init__(
            optimizer,
            base_lr,
        )
        self.reach_base_lr_cost_epoch = reach_base_lr_cost_epoch

        self.lr_list = []
        self.MAX_TRAIN_EPOCH = 100000

    def warm(
            self,
            now_epoch_ind,
            now_batch_ind,
            max_batch_ind
    ):

        if len(self.lr_list) == 0:
            reach_base_lr_cost_batch = self.reach_base_lr_cost_epoch * max_batch_ind
            zero_to_reach_base_lr = np.linspace(0, np.pi / 2, reach_base_lr_cost_batch)

            one_circle = zero_to_reach_base_lr.tolist() \
                         + (np.pi / 2 + zero_to_reach_base_lr).tolist() \
                         + (np.pi + zero_to_reach_base_lr).tolist() \
                         + (np.pi * 3 / 2 + zero_to_reach_base_lr).tolist()

            repeat_num = self.MAX_TRAIN_EPOCH * max_batch_ind // (4 * reach_base_lr_cost_batch) + 1

            total_circle = one_circle * repeat_num

            lr_list = np.abs(np.sin(total_circle)) * self.base_lr
            self.lr_list = lr_list.tolist()

        self.tmp_lr = self.lr_list[now_epoch_ind * max_batch_ind + now_batch_ind]

        self.set_lr(self.tmp_lr)
