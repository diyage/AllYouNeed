from Package.BaseDev.predictor import BasePredictor
import torch
from abc import abstractmethod


class DevPredictor(BasePredictor):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def decode_predict(
            self,
            predict: dict,
            *args,
            **kwargs
    ) -> dict:
        pass

    @abstractmethod
    def decode_pre_feature(
            self,
            pre_feature: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        pass

    def decode_target(
            self,
            target: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        return target
