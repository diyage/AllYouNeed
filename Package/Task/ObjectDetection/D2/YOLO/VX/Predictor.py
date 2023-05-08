# 和之前的Predictor不同，YOLO vx Predictor需要做的东西很少，因为在YOLO vx Model中已经涉及了box解码
# 而box的 nms  是需要Predictor来做的
# 因此outputs需要使用Predictor来做一些后处理，targets 直接丢弃解码操作（因为这不是必须的，直接从DataLoader获取即可），


from Package.Task.ObjectDetection.D2.YOLO.VX.Typing import *
from Package.Task.ObjectDetection.D2.YOLO.VX.Model import YOLOVXModel
from Package.BaseDev.predictor import BasePredictor
import torch
import numpy as np


class YOLOVXPredictor(BasePredictor):
    def __init__(
            self,
            model: YOLOVXModel,
            iou_th: float,
            conf_th: float,
            score_th: float,
            image_size: int = 640
    ):
        super().__init__()
        self.model = model
        self.iou_th = iou_th
        self.conf_th = conf_th
        self.score_th = score_th
        self.image_size = image_size

    def decode_predict(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError('This method has been discarded!')

    def decode_target(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError('This method has been discarded!')

    def nms_iou(
            self,
            box: torch.Tensor,
            score: torch.Tensor
    ) -> Union[torch.Tensor, List[int]]:
        position_abs = box.cpu().detach().numpy().copy()
        scores = score.cpu().detach().numpy().copy()
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
            inds = np.where(ovr <= self.iou_th)[0]
            order = order[inds + 1]
        return keep

    def __process_one_predict(
            self,
            output: torch.Tensor
    ) -> KPS_VEC:

        position = output[..., :4].clamp(0, self.image_size - 1)  # shape [box_num, 4]
        conf = torch.sigmoid(output[..., 4])  # shape [box_num, ]
        cls = torch.sigmoid(output[..., 5:])  # shape [box_num, cls_num]

        _, cls_num = cls.shape
        score = cls * conf.unsqueeze(dim=-1)
        scores_max_value, scores_max_index = score.max(dim=-1)  # shape [box_num, ]

        th_mask = (conf > self.conf_th) & (scores_max_value > self.score_th)
        res = []
        for cls_ind in range(cls_num):
            cls_mask = scores_max_index == cls_ind  # shape [box_num, ]
            box_mask = th_mask & cls_mask

            position_ = position[box_mask]
            # score_ = score[box_mask]
            # be careful here, we just need score max value not the natural score
            scores_max_value_ = scores_max_value[box_mask]

            keep_ind = self.nms_iou(
                position_,
                scores_max_value_
            )
            res += [
                (
                    cls_ind,
                    tuple(position_[ind].cpu().detach().numpy().tolist()),
                    scores_max_value_[ind].item()
                ) for ind in keep_ind
            ]

        return res

    def post_process(
            self,
            outputs: torch.Tensor
    ) -> List[KPS_VEC]:
        assert len(outputs.shape) == 3
        batch_size = outputs.shape[0]
        res = []
        for batch_ind in range(batch_size):
            res.append(
                self.__process_one_predict(outputs[batch_ind])
            )
        return res
