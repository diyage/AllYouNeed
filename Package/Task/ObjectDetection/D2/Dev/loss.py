from abc import abstractmethod
from Package.BaseDev import BaseLoss


class DevLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        """
        Returns:
            {
                'total_loss': xxx,
                'loss_key_1': xxx,
                'loss_key_2': xxx,
                ...
            }
        """
        pass
