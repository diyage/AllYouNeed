import torch
import math


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
            self.tmp_lr = self.base_lr * pow((now_batch_ind + now_epoch_ind * max_batch_ind) * 1. / (self.warm_up_epoch * max_batch_ind), 4)
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
