"""
This packet(predictor) is core of Object Detection.
It will be used in inference phase for decoding ground-truth(target)/model-output(predict).

You must know some special definitions in my frame:
    kps_vec_s --> [kps_vec0, kps_vec1, ...]                         for batch images
        kps_vec --> [kps0, kps1, kps2, ...]                         for one image
            kps --> (predict_kind_name, abs_double_pos, score)      for one object
                predict_kind_name --> str  (e.g. 'cat', 'dog', 'car', ...)
                abs_double_pos --> (x, y, x, y)   scaled on image
                score --> float   conf * cls_prob
"""
import torch
from typing import Union
from abc import abstractmethod
from typing import List
from Package.BaseDev import BasePredictor
import numpy as np


class DevPredictor(BasePredictor):
    def __init__(
            self,
            iou_th: float,
            prob_th: float,
            conf_th: float,
            score_th: float,
            pre_anchor_w_h_rate: Union[tuple, dict],
            kinds_name: list,
            image_size: tuple,
            image_shrink_rate: Union[tuple, dict]
    ):
        super().__init__()
        self.iou_th = iou_th
        self.prob_th = prob_th
        self.conf_th = conf_th
        self.score_th = score_th
        self.kinds_name = kinds_name

        self.pre_anchor_w_h_rate = pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = image_shrink_rate
        self.grid_number = None

        self.image_size = None
        self.change_image_wh(image_size)

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    @staticmethod
    def single_nms(
            position_abs: torch.Tensor,
            scores: torch.Tensor,
            threshold: float = 0.5
    ):
        position_abs = position_abs.cpu().detach().numpy().copy()
        scores = scores.cpu().detach().numpy().copy()
        """"Pure Python NMS baseline."""
        x1 = position_abs[:, 0]  # xmin
        y1 = position_abs[:, 1]  # ymin
        x2 = position_abs[:, 2]  # xmax
        y2 = position_abs[:, 3]  # ymax

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
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def nms(
            self,
            position_abs_: torch.Tensor,
            scores_max_value: torch.Tensor,
            scores_max_index: torch.Tensor
    ) -> List:
        """
        for one image, all predicted objects( already mask some bad ones)
        it may have many kinds...
        Args:
            position_abs_: (P, 4)
            scores_max_value: (P, ) predicted kind_name's score
            scores_max_index: (P, ) predicted kind_name's index

        Returns: kps_vec

        """
        def for_response(
                now_kind_pos_abs,
                now_kind_scores_max_value,
        ):
            res = []
            keep_index = self.single_nms(
                now_kind_pos_abs,
                now_kind_scores_max_value,
                threshold=iou_th,
            )

            for index in keep_index:
                s = now_kind_scores_max_value[index]
                abs_double_pos = tuple(now_kind_pos_abs[index].cpu().detach().numpy().tolist())
                predict_kind_name = kind_name

                res.append(
                    (predict_kind_name, abs_double_pos, s.item())  # kps
                )

            return res

        iou_th = self.iou_th
        kinds_name = self.kinds_name

        total = []
        for kind_index, kind_name in enumerate(kinds_name):
            now_kind_response = scores_max_index == kind_index
            total = total + for_response(
                position_abs_[now_kind_response],
                scores_max_value[now_kind_response],
            )

        return total

    @abstractmethod
    def decode_one_target(
            self,
            *args,
            **kwargs
    ) -> List:
        """
        Returns:
            kps_vec
        """
        pass

    @abstractmethod
    def decode_target(
            self,
            *args,
            **kwargs
    ) -> List[List]:
        """
        Returns:
                    kps_vec_s
        """
        pass

    @abstractmethod
    def decode_one_predict(
            self,
            *args,
            **kwargs
    ) -> List:
        """
        Returns:
                            kps_vec
        """
        pass

    @abstractmethod
    def decode_predict(
            self,
            *args,
            **kwargs
    ) -> List[List]:
        """
        Returns:
                            kps_vec_s
        """
        pass

