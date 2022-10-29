"""
This packet is not important.
But I suggest --your loss func/class-- inherit from BaseLoss.
And please overwrite --forward-- method.
"""
from abc import abstractmethod
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        pass
