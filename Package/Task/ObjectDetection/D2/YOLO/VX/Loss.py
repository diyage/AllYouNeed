# 和模型结构一样，损失函数的计算绝对是了解深度学习任务的核心
# 可惜的是很多开源代码里都没有给出损失函数的计算，仅仅是推理阶段

from Package.Task.ObjectDetection.D2.Dev import DevLoss
from Package.Task.ObjectDetection.D2.YOLO.VX.Tools import YOLOVXTool
from typing import *
import torch
import torch.nn as nn


class YOLOVXLoss(DevLoss):
    def __init__(
            self,
            weight_position: float = 1.0,
            weight_conf_has_obj: float = 1.0,
            weight_conf_no_obj: float = 1.0,
            weight_cls_prob: float = 1.0,
    ):
        super().__init__()
        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_cls_prob = weight_cls_prob
        self.bce = nn.BCELoss()

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        raise RuntimeError("This method has been discarded!")

    @staticmethod
    def get_position_loss(
            pre_position: torch.Tensor,
            gt_position: torch.Tensor
    ) -> torch.Tensor:
        r"""
        注意：
            pre_position 中有很多 坐标是不符合 取值范围的（因为在model的解码阶段，我并没有采用clip操作）
            所以这个c_iou的计算要相当小心，稍不注意可能导致none损失的产生
        """
        c_iou = YOLOVXTool.c_iou(pre_position, gt_position)
        c_iou_loss = (1.0 - c_iou).mean()
        return c_iou_loss

    def get_no_obj_loss(
            self,
            pre_conf: torch.Tensor,
            gt_conf: torch.Tensor
    ) -> torch.Tensor:
        return self.bce(pre_conf, gt_conf)

    def get_has_obj_loss(
            self,
            pre_conf: torch.Tensor,
            gt_conf: torch.Tensor
    ) -> torch.Tensor:
        return self.bce(pre_conf, gt_conf)

    def get_cls_loss(
            self,
            pre_cls: torch.Tensor,
            gt_cls: torch.Tensor
    ) -> torch.Tensor:
        return self.bce(pre_cls, gt_cls)

    def forward(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, float]]:
        r"""
        该方法负责计算损失函数
            outputs是模型的输出，形状为 [batch, box_num, info_num]
            targets和outputs保持一致

        """
        res = {
            'total_loss': 0.0,
            'position_loss': 0.0,
            'no_obj_loss': 0.0,
            'has_obj_loss': 0.0,
            'cls_loss': 0.0,
        }
        gt_position = targets[..., :4]   # shape[batch, box_num, 4]
        gt_conf = targets[..., 4]  # shape[batch, box_num]
        gt_cls = targets[..., 5:]  # shape[batch, box_num, cls_num]

        pre_position = outputs[..., :4]  # shape[batch, box_num, 4]
        pre_conf = outputs[..., 4]  # shape[batch, box_num]
        pre_cls = outputs[..., 5:]  # shape[batch, box_num, cls_num]

        mask_has_obj = gt_conf == 1.0  # shape[batch, box_num]
        mask_no_obj = ~mask_has_obj

        res['position_loss'] = self.get_position_loss(
            pre_position[mask_has_obj],
            gt_position[mask_has_obj]
        )
        res['no_obj_loss'] = self.get_no_obj_loss(
            pre_conf[mask_no_obj],
            gt_conf[mask_no_obj]
        )
        res['has_obj_loss'] = self.get_has_obj_loss(
            pre_conf[mask_has_obj],
            gt_conf[mask_has_obj]
        )
        res['cls_loss'] = self.get_cls_loss(
            pre_cls[mask_has_obj],
            gt_cls[mask_has_obj]
        )
        res['total_loss'] = self.weight_position * res['position_loss'] + \
            self.weight_conf_no_obj * res['no_obj_loss'] + \
            self.weight_conf_has_obj * res['has_obj_loss'] + \
            self.weight_cls_prob * res['cls_loss']
        return res
