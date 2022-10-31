from Package.BaseDev.model import BaseModel
import torch.nn as nn
from abc import abstractmethod


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
    ):
        """
        :return  one dict
            {
                "feature": xxx,
                "out": xxx,
            }
        """
        pass
