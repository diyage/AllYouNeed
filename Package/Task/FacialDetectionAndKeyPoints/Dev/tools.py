from Package.BaseDev import BaseTool
from abc import abstractmethod
import torch
import numpy as np


class DevTool(BaseTool):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def make_target(
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement make_target for xxTool"
        )

    @staticmethod
    @abstractmethod
    def split_target(
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement split_target for xxTool"
        )

    @staticmethod
    @abstractmethod
    def split_predict(
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement split_target for xxTool"
        )

    @staticmethod
    def single_kind_nms(
            position: torch.Tensor,
            scores: torch.Tensor,
            threshold: float = 0.5
    ):
        position = position.cpu().detach().numpy().copy()
        scores = scores.cpu().detach().numpy().copy()
        """"Pure Python NMS baseline."""
        x1 = position[:, 0]  # xmin
        y1 = position[:, 1]  # ymin
        x2 = position[:, 2]  # xmax
        y2 = position[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-20)
            # 滤除超过nms阈值的检测框
            ind_s = np.where(ovr <= threshold)[0]
            order = order[ind_s + 1]

        return keep
