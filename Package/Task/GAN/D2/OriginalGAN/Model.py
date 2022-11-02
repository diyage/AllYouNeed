from ..Dev import DevModel
import torch.nn as nn
import torch


class OriginalGANModel(DevModel):
    def __init__(
            self,
            g_net: nn.Module,
            d_net: nn.Module
    ):
        super().__init__(g_net, d_net)

    def get_fake_images(
            self,
            now_batch_size: int,
            noise_channel: int,
            *args,
            **kwargs
    ) -> torch.Tensor:
        device = next(self.generator.parameters()).device
        random_noise = torch.randn(
            size=(now_batch_size, noise_channel, 1, 1)
        ).to(device)
        fake_images: torch.Tensor = self.generator(random_noise)
        return fake_images

    def get_predict(
            self,
            images: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        return self.discriminator(images)
