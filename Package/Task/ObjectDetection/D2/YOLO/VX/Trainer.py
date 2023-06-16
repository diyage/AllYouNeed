# 可以发现，在之前的Trainer，我都是从DevTrainer继承过来，DevTrainer中参数太复杂了，违背了AllYouNeed设计的初衷（一切由程序员掌控）
# 控制权，我希望是无穷大
# 因此，在YOLO X等后续的代码中
# 我都选择从BaseTrainer继承
from torch.utils.data import DataLoader
from Package.Task.ObjectDetection.D2.YOLO.VX.Typing import *
from Package.BaseDev import BaseTrainer
from Package.Task.ObjectDetection.D2.YOLO.VX.Loss import YOLOVXLoss
from Package.Task.ObjectDetection.D2.YOLO.VX.Model import YOLOVXModel
from Package.Task.ObjectDetection.D2.YOLO.VX.Tools import YOLOVXTool
from Package.Optimizer.WarmUp import WarmUpOptimizer, WarmUpCosineAnnealOptimizer, WarmUpAbsSineCircleOptimizer
from tqdm import tqdm
import torch


class YOLOVXTrainer(BaseTrainer):
    def __init__(
            self,
            model: YOLOVXModel,
            lr_mapping: Dict[int, Dict[int, float]],
            width_height_center,
            image_size: int = 640,
            image_shrink_rate: Tuple[int, int, int] = (8, 16, 32),
            cls_num: int = 80,
            multi_positives: bool = True
    ):
        super().__init__()
        self.model = model
        self.width_height_center = width_height_center
        self.lr_mapping = lr_mapping
        self.device = next(model.parameters()).device
        self.image_size = image_size
        self.image_shrink_rate = image_shrink_rate
        self.cls_num = cls_num
        self.multi_positives = multi_positives

    def change_lr_mapping(
            self,
            new_lr_mapping: Dict[int, Dict[int, float]],
    ):
        self.lr_mapping = new_lr_mapping

    def adjust_lr(
            self,
            optimizer: Union[WarmUpOptimizer, WarmUpCosineAnnealOptimizer, WarmUpAbsSineCircleOptimizer],
            now_epoch: int,
            now_batch: int
    ):
        if self.lr_mapping.get(now_epoch) is None or self.lr_mapping[now_epoch].get(now_batch) is None:
            r"""
            沿用之前的lr
            """
            pass
        else:
            optimizer.set_lr(self.lr_mapping[now_epoch][now_batch])

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: YOLOVXLoss,
            optimizer: Union[WarmUpOptimizer, WarmUpCosineAnnealOptimizer, WarmUpAbsSineCircleOptimizer],
            desc: str = '',
            now_epoch: int = 0,
    ) -> dict:
        r"""*
        这是 trainer 的核心，早期的代码里，总是希望optimizer自带lr的调节功能，发现总是不太令人满意，
        最终，我选择将optimizer的学习率策略交给使用者，设置一个lr_mapping，把学习率的策略精确到batch级别，
        细粒度，这可能会对使用者要求较高，但是我比较喜欢
        """
        loss_dict_vec = {}
        print_frequency = max(1, int(len(data_loader_train) * 0.1))
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            self.adjust_lr(
                optimizer,
                now_epoch,
                batch_id
            )
            self.model.train()
            images = images.to(self.device)

            outputs = self.model(images)

            # targets = YOLOVXTool.make_target(
            #     labels,
            #     self.width_height_center,
            #     self.image_size,
            #     self.image_shrink_rate,
            #     self.cls_num,
            #     self.multi_positives
            # ).to(self.device)

            targets = YOLOVXTool.sim_ota_make_target(
                outputs,
                labels,
                self.image_size,
                self.image_shrink_rate,
            ).to(self.device)

            loss_res = loss_func(outputs, targets)

            # outputs = self.model(images)
            #
            # torch.cuda.empty_cache()
            # targets = YOLOVXTool.sim_ota_make_target(
            #     outputs,
            #     labels,
            #     self.image_size,
            #     self.image_shrink_rate,
            #
            # ).to(self.device)
            # torch.cuda.empty_cache()

            # loss_res = loss_func(outputs, targets)

            if not isinstance(loss_res, dict):
                raise RuntimeError(
                    'You have not use our provided loss func, please overwrite method train_detector_one_epoch'
                )
            else:
                if batch_id % print_frequency == 0:
                    print()
                    print("epoch: {}, batch: {}".format(now_epoch, batch_id))
                    for k, v in loss_res.items():
                        print("{}: {}".format(k, v))
                    print()

                loss = loss_res['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for key, val in loss_res.items():
                    if key not in loss_dict_vec.keys():
                        loss_dict_vec[key] = []
                    loss_dict_vec[key].append(val.item())

        loss_dict = {}
        for key, val in loss_dict_vec.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict

