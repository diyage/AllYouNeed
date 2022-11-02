from ..Dev import DevLoss
import torch


class WGANLoss(DevLoss):
    def __init__(
            self,

    ):
        super().__init__()

    def compute_discriminator_loss(
            self,
            real_predict: torch.Tensor,
            fake_predict: torch.Tensor,
            *args,
            **kwargs
    ) -> dict:

        loss = -torch.mean(real_predict) + torch.mean(fake_predict)

        return {
            'total_loss': loss,
        }

    def compute_generator_loss(
            self,
            fake_predict: torch.Tensor,
            *args,
            **kwargs
    ) -> dict:

        loss = -torch.mean(fake_predict)
        return {
            'total_loss': loss
        }
