from ..Dev import DevLoss
from .Tools import OriginalGANTool
import torch
import torch.nn as nn


class OriginalGANLoss(DevLoss):
    def __init__(
            self,

    ):
        super().__init__()
        self.bce_loss_func = nn.BCELoss()

    def compute_discriminator_loss(
            self,
            real_predict: torch.Tensor,
            fake_predict: torch.Tensor,
            *args,
            **kwargs
    ) -> dict:
        real_target: torch.Tensor = OriginalGANTool.make_target(
            real_predict.shape[0],
            is_real_image=True
        ).to(real_predict.device)
        fake_target: torch.Tensor = OriginalGANTool.make_target(
            fake_predict.shape[0],
            is_real_image=False
        ).to(fake_predict.device)
        d_real_loss = self.bce_loss_func(real_predict, real_target)
        d_fake_loss = self.bce_loss_func(fake_predict, fake_target)
        loss = d_real_loss + d_fake_loss

        return {
            'total_loss': loss,
            'd_real_loss': d_real_loss,
            'd_fake_loss': d_fake_loss
        }

    def compute_generator_loss(
            self,
            fake_predict: torch.Tensor,
            *args,
            **kwargs
    ) -> dict:
        fake_target: torch.Tensor = OriginalGANTool.make_target(
            fake_predict.shape[0],
            is_real_image=True
        ).to(fake_predict.device)
        loss = self.bce_loss_func(fake_predict, fake_target)
        return {
            'total_loss': loss
        }
