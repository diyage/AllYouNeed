import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Package.Task.ObjectDetection.D2.YOLO.V5 import *
from Package.Optimizer.WarmUp import WarmUpOptimizer, WarmUpAbsSineCircleOptimizer
from Demo.yolo_v5.other.config import YOLOV5Config
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLOV5Helper:
    def __init__(
            self,
            model: YOLOV5Model,
            config: YOLOV5Config,
            restore_epoch: int = -1
    ):
        self.model = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = YOLOV5Trainer(
            model,
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.kinds_name,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

        self.predictor_for_show = YOLOV5Predictor(
            self.config.show_config.iou_th_for_show,
            self.config.show_config.prob_th_for_show,
            self.config.show_config.conf_th_for_show,
            self.config.show_config.score_th_for_show,
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.single_an
        )

        self.predictor_for_eval = YOLOV5Predictor(
            self.config.eval_config.iou_th_for_eval,
            self.config.eval_config.prob_th_for_eval,
            self.config.eval_config.conf_th_for_eval,
            self.config.eval_config.score_th_for_eval,
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.single_an
        )

        self.visualizer = YOLOV5Visualizer(
            model,
            self.predictor_for_show,
            self.config.data_config.class_colors,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

        self.my_evaluator = YOLOV5Evaluator(
            model,
            self.predictor_for_eval,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

    def restore(
            self,
            epoch: int
    ):
        self.restore_epoch = epoch
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
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
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
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

    def eval_map(
            self,
            data_loader_test: DataLoader,
    ):
        with torch.no_grad():
            self.my_evaluator.eval_map(
                data_loader_test,
                desc='[eval detector mAP]',
                use_07_metric=self.config.eval_config.use_07_metric
            )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOV5Loss(
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.single_an,
            self.config.train_config.weight_position,
            self.config.train_config.weight_conf_has_obj,
            self.config.train_config.weight_conf_no_obj,
            self.config.train_config.weight_cls_prob,
            image_size=self.config.data_config.image_size,
        )

        sgd_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.train_config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        warm_optimizer = WarmUpOptimizer(
            sgd_optimizer,
            base_lr=self.config.train_config.lr,
            warm_up_epoch=self.config.train_config.reach_base_lr_cost_epoch
        )

        for epoch in tqdm(range(self.restore_epoch+1,
                                self.config.train_config.max_epoch_on_detector),
                          desc='training object detector',
                          position=0):
            if epoch in [50, 100, 150]:
                warm_optimizer.set_lr(warm_optimizer.tmp_lr * 0.1)
            loss_dict = self.trainer.train_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                desc='train for detector epoch --> {}'.format(epoch),
                now_epoch=epoch,

            )

            print_info = '\n\nepoch: {} [ now lr:{:.6f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            self.save(epoch)
            if epoch % self.config.eval_config.eval_frequency == 0 and epoch != 0:
                self.show_detect_results(data_loader_test, epoch)
                self.eval_map(data_loader_test)
