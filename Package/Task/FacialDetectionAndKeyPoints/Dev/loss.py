from Package.BaseDev import BaseLoss
import abc


class DevLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement this method forward for xxLoss"
        )
