"""
from Package.BaseDev import BaseModel
be careful, in fact, we need inherit BaseModel, but we do not chose do that.
"""
import torch.nn as nn
from abc import abstractmethod


class DevModel(nn.Module):
    def __init__(
            self,
            g_net: nn.Module,
            d_net: nn.Module
    ):
        super().__init__()
        self.generator: nn.Module = g_net
        self.discriminator: nn.Module = d_net

    @abstractmethod
    def get_fake_images(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            'Please implement method get_fake_images(for xxModel).'
        )

    @abstractmethod
    def get_predict(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            'Please implement method get_predict(for xxModel).'
        )

    def forward(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            "Do not use __call__, the situation of GAN's Module is complex! " +
            "Please use methods get_predict and get_fake_images."
        )
