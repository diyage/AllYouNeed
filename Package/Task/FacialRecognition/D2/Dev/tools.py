from Package.BaseDev.tools import BaseTool
from abc import abstractmethod
import torch


class DevTool(BaseTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def make_target(
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def split_predict(
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def split_target(
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def l2_norm(
            x: torch.Tensor,
            dim: int,
    ) -> torch.Tensor:
        res = x / torch.norm(x, 2, dim=dim, keepdim=True)
        return res
