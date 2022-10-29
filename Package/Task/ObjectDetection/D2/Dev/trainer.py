import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union
from abc import abstractmethod
from .model import DevModel
from Package.BaseDev import BaseTrainer


class DevTrainer(BaseTrainer):
    def __init__(
            self,
            model: DevModel,
            pre_anchor_w_h_rate: Union[tuple, dict],
            image_size: tuple,
            image_shrink_rate: Union[tuple, dict],
            kinds_name: list,
            iou_th_for_make_target: float
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

        self.pre_anchor_w_h_rate = pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = image_shrink_rate
        self.grid_number = None

        self.image_size = None
        self.change_image_wh(image_size)

        self.kinds_name = kinds_name

        self.iou_th_for_make_target = iou_th_for_make_target

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        pass

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: nn.Module,
            optimizer: torch.optim.Optimizer,
            desc: str = '',
    ):
        loss_dict_vec = {}

        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):

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


