import torch.nn as nn
import abc


class DevModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement this method forward for xxModel."
        )
