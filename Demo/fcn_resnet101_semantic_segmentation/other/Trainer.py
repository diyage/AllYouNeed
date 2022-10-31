from Package.Task.Segmentation.SemanticSegmentation.D2.Dev import DevTrainer, DevTool
from .Model import FCNResNet101Model
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Union, List
import numpy as np
from Package.Optimizer.WarmUp import WarmUpCosineAnnealOptimizer, WarmUpOptimizer
from tqdm import tqdm


class FCNResNet101Trainer(DevTrainer):
    def __init__(
            self,
            model: FCNResNet101Model
    ):
        super().__init__(model)

    def make_targets(
            self,
            labels: List[np.ndarray],
            *args,
            **kwargs
    ):
        targets = DevTool.make_target(
            labels,
        )
        return targets.to(self.device)

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: nn.Module,
            optimizer: Union[WarmUpOptimizer, WarmUpCosineAnnealOptimizer],
            desc: str = '',
            now_epoch: int = 0,
    ):
        loss_dict_vec = {}
        max_batch_ind = len(data_loader_train)
        for batch_id, (images, objects, labels) in enumerate(tqdm(data_loader_train,
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
