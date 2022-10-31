"""
This packet is the most-most-most core development tool.
It will serve for all other development tools.
You could use it to define everything !!!
"""
import torch
import numpy as np
from Package.BaseDev import BaseTool
from typing import List


class DevTool(BaseTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    def make_target(
            labels: List[np.ndarray],
            *args,
            **kwargs
    ) -> torch.Tensor:
        """

        :param labels:  has saved many masks(batch_size * mask_num * H * W)
        :param args:
        :param kwargs:
        :return:
        """
        return torch.from_numpy(np.array(labels))

    @staticmethod
    def split_target(
            target: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        just swap dimensions. (from n*c*h*w to n*h*w*c)
        :param target: batch_size * mask_num * h * w
        :param args:
        :param kwargs:
        :return:
        """
        return target.permute(0, 2, 3, 1)

    @staticmethod
    def split_predict(
            predict: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """
        just swap dimensions.  (from n*c*h*w to n*h*w*c)
        :param predict: batch_size * mask_num * h * w
        :param args:
        :param kwargs:
        :return:
        """
        return predict.permute(0, 2, 3, 1)
