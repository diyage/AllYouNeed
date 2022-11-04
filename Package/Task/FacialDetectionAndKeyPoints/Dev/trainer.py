from Package.BaseDev import BaseTrainer
import abc


class DevTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def train_one_epoch(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement this method train_one_epoch for xxTrainer."
        )
