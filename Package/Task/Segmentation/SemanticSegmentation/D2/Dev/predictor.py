import torch
from .tools import DevTool
import torch.nn.functional as F
from Package.BaseDev import BasePredictor
import numpy as np


class DevPredictor(BasePredictor):
    def __init__(
            self,
    ):
        super().__init__()

    def decode_target(
            self,
            target: torch.Tensor,
    ) -> np.ndarray:
        target = DevTool.split_target(target)

        return target.cpu().detach().numpy()

    def decode_predict(
            self,
            predict: torch.Tensor,
    ) -> np.ndarray:
        predict = DevTool.split_predict(predict)
        pre_mask_vec = F.one_hot(predict.argmax(dim=-1), num_classes=predict.shape[-1])
        return pre_mask_vec.cpu().detach().numpy()
