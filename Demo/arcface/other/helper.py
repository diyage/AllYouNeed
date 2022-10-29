from Package.Task.FacialRecognition.D2.ArcFace import *
from Package.Optimizer.WarmUp import WarmUpCosineAnnealOptimizer
from .config import ArcFaceConfig
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List


class ArcFaceHelper:
    def __init__(
            self,
            model: ArcFaceModel,
            config: ArcFaceConfig,
            restore_epoch: int = -1
    ):
        self.model = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = ArcFaceTrainer(
            model
        )
        self.predictor_for_eval = ArcFacePredictor(
            use_l2_norm_feature=self.config.use_l2_norm_feature
        )
        self.my_evaluator = ArcFaceEvaluator(
            model,
            self.predictor_for_eval,
            compute_threshold_num=self.config.compute_threshold_num,
            distance_type=self.config.distance_type
        )

    def restore(
            self,
            epoch: int
    ):
        self.restore_epoch = epoch
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth/'
        saved_file_name = '{}/{}.pth'.format(saved_dir, epoch)
        self.model.load_state_dict(
            torch.load(saved_file_name)
        )

    def save(
            self,
            epoch: int
    ):
        # save model
        self.model.eval()
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.model.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

    def eval_acc(
            self,
            data_loader_test: DataLoader,
            data_pair: List[List]
    ):
        with torch.no_grad():
            self.my_evaluator.eval_verification_accuracy(
                data_loader_test,
                data_pair=data_pair,
                desc='[eval accuracy]',
            )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
            data_pair: List[List]
    ):
        loss_func = ArcFaceLoss()

        sgd_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        warm_optimizer = WarmUpCosineAnnealOptimizer(
            sgd_optimizer,
            max_epoch_for_train=self.config.max_epoch_for_train,
            base_lr=self.config.lr,
            warm_up_end_epoch=self.config.warm_up_end_epoch
        )

        for epoch in tqdm(range(self.restore_epoch+1,
                                self.config.max_epoch_for_train),
                          desc='training',
                          position=0):

            # if epoch in [50, 100, 150]:
            #     warm_optimizer.set_lr(warm_optimizer.tmp_lr * 0.1)

            loss_dict = self.trainer.train_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                desc='train for epoch --> {}'.format(epoch),
                now_epoch=epoch,
            )

            print_info = '\n\nepoch: {} [ now lr:{:.6f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if epoch % self.config.eval_frequency == 0:
                self.save(epoch)
                self.eval_acc(data_loader_test, data_pair)
