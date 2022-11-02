from Package.BaseDev import BaseLoss
from abc import abstractmethod


class DevLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_generator_loss(
            self,
            *args,
            **kwargs
    ) -> dict:
        raise NotImplementedError(
            'Please implement method compute_generator_loss(for xxLoss).'
        )

    @abstractmethod
    def compute_discriminator_loss(
            self,
            *args,
            **kwargs
    ) -> dict:
        raise NotImplementedError(
            'Please implement method compute_discriminator_loss(for xxLoss).'
        )

    def forward(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            "Do not use __call__, the situation of GAN's Loss is complex! " +
            "Please use methods get_predict and get_fake_images."
        )
