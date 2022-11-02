from Package.BaseDev import BaseTrainer
from abc import abstractmethod


class DevTrainer(BaseTrainer):
    def __init__(
            self,
            train_g_frequency: int = 1,
            train_d_frequency: int = 1,
    ):
        super().__init__()
        self.train_g_frequency = train_g_frequency
        self.train_d_frequency = train_d_frequency

    @abstractmethod
    def train_one_epoch(
            self,
            *args,
            **kwargs
    ) -> dict:
        """
        train the model, and return some useful information.
        """
        raise NotImplementedError(
            "Please implement method train_one_epoch(for xxTrainer)."
        )
