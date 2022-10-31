from Package.BaseDev.trainer import BaseTrainer
from Package.Task.FacialRecognition.D2.Dev.model import DevModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from abc import abstractmethod


class DevTrainer(BaseTrainer):
    def __init__(
            self,
            model: DevModel
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

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
            """
            be careful, output is a dict.
            """
            loss_res = loss_func(output, targets)
            if not isinstance(loss_res, dict):
                print('You have not use our provided loss func...')
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
