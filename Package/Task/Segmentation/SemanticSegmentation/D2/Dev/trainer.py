import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import numpy as np
from abc import abstractmethod
from .model import DevModel
from Package.BaseDev import BaseTrainer


class DevTrainer(BaseTrainer):
    def __init__(
            self,
            model: DevModel,
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    @abstractmethod
    def make_targets(
            self,
            labels: List[np.ndarray],
            *args,
            **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: nn.Module,
            optimizer: torch.optim.Optimizer,
            desc: str = '',
    ):
        pass
