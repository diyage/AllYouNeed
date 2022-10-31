from Package.BaseDev.loss import BaseLoss
from abc import abstractmethod
import torch


class DevLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
            self,
            predict: dict,
            target: torch.Tensor,
            *args,
            **kwargs
    ):
        pass
