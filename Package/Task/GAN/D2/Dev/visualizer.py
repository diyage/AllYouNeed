from Package.BaseDev import BaseVisualizer
from .model import DevModel
from abc import abstractmethod
from typing import List


class DevVisualizer(BaseVisualizer):
    def __init__(
            self,
            model: DevModel,
            mean: List[float],
            std: List[float],
    ):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    @abstractmethod
    def show_generate_results(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            'Please implement method show_generate_results(for xxVisualizer).'
        )


