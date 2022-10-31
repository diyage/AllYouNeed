from Package.Task.Segmentation.SemanticSegmentation.D2.Dev import DevPredictor as FCNResNet101Predictor
from Package.Task.Segmentation.SemanticSegmentation.D2.Dev import DevEvaluator as FCNResNet101Evaluator
from Package.Task.Segmentation.SemanticSegmentation.D2.Dev import DevVisualizer as FCNResNet101Visualizer
from .config import FCNResNet101Config
from .Loss import FCNResNet101Loss
from .Model import FCNResNet101Model
from .Trainer import FCNResNet101Trainer
from Package.Optimizer.WarmUp import WarmUpCosineAnnealOptimizer
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import DataLoader


class FCNResNet101Helper:
    def __init__(
            self,
            model: FCNResNet101Model,
            config: FCNResNet101Config,
            restore_epoch: int = -1
    ):
        self.model = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = FCNResNet101Trainer(
            model,
        )

        self.predictor_for_show = FCNResNet101Predictor()

        self.predictor_for_eval = FCNResNet101Predictor()

        self.visualizer = FCNResNet101Visualizer(
            model,
            self.predictor_for_show,
            image_mean=self.config.data_config.mean,
            image_std=self.config.data_config.std,
            kind_name_to_color=self.config.data_config.KIND_NAME_TO_COLOR
        )

        self.my_evaluator = FCNResNet101Evaluator(
            model,
            self.predictor_for_eval,
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

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            epoch: int
    ):
        with torch.no_grad():
            saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.visualizer.show_detect_results(
                data_loader_test,
                saved_dir,
                desc='[show predict results]'
            )

    def eval_semantic_segmentation_accuracy(
            self,
            data_loader_test: DataLoader,
    ):
        with torch.no_grad():
            self.my_evaluator.eval_semantic_segmentation_accuracy(
                data_loader_test,
                desc='[eval semantic segmentation accuracy]'
            )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = FCNResNet101Loss()

        sgd_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.train_config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        warm_optimizer = WarmUpCosineAnnealOptimizer(
            sgd_optimizer,
            self.config.train_config.max_epoch_on_detector,
            base_lr=self.config.train_config.lr,
            warm_up_end_epoch=self.config.train_config.warm_up_end_epoch
        )

        for epoch in tqdm(range(self.restore_epoch + 1, self.config.train_config.max_epoch_on_detector),
                          desc='training detector',
                          position=0):

            loss_dict = self.trainer.train_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                desc='[train for detector epoch: {}/{}]'.format(epoch,
                                                                self.config.train_config.max_epoch_on_detector - 1),
                now_epoch=epoch

            )

            print_info = '\n\nepoch: {} [ now lr:{:.8f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if epoch % self.config.eval_config.eval_frequency == 0:
                # save model
                self.save(epoch)

                # show predict
                self.show_detect_results(data_loader_test, epoch)

                # eval accuracy
                self.eval_semantic_segmentation_accuracy(data_loader_test)
