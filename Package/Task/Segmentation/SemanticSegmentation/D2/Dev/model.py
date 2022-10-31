from abc import abstractmethod
import torch.nn as nn
import torch
from Package.BaseDev import BaseModel


class DevModel(BaseModel):
    def __init__(
            self,
            net: nn.Module
    ):
        super().__init__(net)

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ) -> torch.Tensor:
        pass
