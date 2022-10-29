"""
This packet is not important.
"""
from abc import abstractmethod


class BaseTrainer:
    def __init__(
            self,
    ):
        pass

    @abstractmethod
    def train_one_epoch(
            self,
            *args,
            **kwargs
    ) -> dict:
        """
        train the model, and return some useful information.
        """
        pass






