"""
This packet is not important.
And its function equals to --make_predict-- .
"""
from abc import abstractmethod
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
            self,
            net: nn.Module
    ):
        super().__init__()
        self.net = net
        """
        backbone is real model(or net).
        """

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        pass
