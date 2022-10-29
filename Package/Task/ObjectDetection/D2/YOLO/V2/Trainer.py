import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .Tools import YOLOV2Tool
from .Model import YOLOV2Model
from Package.Task.ObjectDetection.D2.Dev import DevTrainer
from Package.Optimizer.WarmUp import WarmUpOptimizer


class YOLOV2Trainer(DevTrainer):
    def __init__(
            self,
            model: YOLOV2Model,
            pre_anchor_w_h_rate: tuple,
            image_size: tuple,
            image_shrink_rate: tuple,
            kinds_name: list,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            pre_anchor_w_h_rate,
            image_size,
            image_shrink_rate,
            kinds_name,
            iou_th_for_make_target
        )

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV2Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels,
    ):
        return YOLOV2Tool.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
        ).to(self.device)

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: nn.Module,
            optimizer: WarmUpOptimizer,
            desc: str = '',
            now_epoch: int = 0,
    ):
        loss_dict_vec = {}
        max_batch_ind = len(data_loader_train)
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            optimizer.warm(
                now_epoch,
                batch_id,
                max_batch_ind
            )
            self.model.train()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            output = self.model(images)
            loss_res = loss_func(output, targets)
            if not isinstance(loss_res, dict):
                print('You have not use our provided loss func, please overwrite method train_detector_one_epoch')
                pass
            else:
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
