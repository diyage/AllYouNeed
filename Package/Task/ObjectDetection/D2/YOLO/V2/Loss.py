import torch
import torch.nn as nn
from Package.Task.ObjectDetection.D2.Dev import DevLoss
from .Tools import YOLOV2Tool


class YOLOV2Loss(DevLoss):
    def __init__(
            self,
            pre_anchor_w_h_rate: tuple,
            weight_position: float = 1.0,
            weight_conf_has_obj: float = 1.0,
            weight_conf_no_obj: float = 1.0,
            weight_cls_prob: float = 1.0,
            weight_iou_loss: float = 1.0,
            image_shrink_rate: tuple = (13, 13),
            image_size: tuple = (416, 416),
    ):
        super().__init__()
        self.anchor_number = len(pre_anchor_w_h_rate)

        self.pre_anchor_w_h_rate = pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = image_shrink_rate
        self.grid_number = None  # type: tuple

        self.image_size = None
        self.change_image_wh(image_size)

        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_cls_prob = weight_cls_prob
        self.weight_iou_loss = weight_iou_loss

        self.mse = nn.MSELoss(reduction='none')
        self.iou_loss_function = nn.SmoothL1Loss(reduction='none')

        self.iteration = 0

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV2Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def txtytwth_xyxy(
            self,
            txtytwth: torch.Tensor,
    ) -> torch.Tensor:
        # offset position to abs position
        return YOLOV2Tool.xywh_to_xyxy(
            txtytwth,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number
        )

    def xyxy_txty_s_twth(
            self,
            xyxy: torch.Tensor,
    ) -> torch.Tensor:

        return YOLOV2Tool.xyxy_to_xywh(
            xyxy,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number
        )

    def forward(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        N = out.shape[0]
        """
        split output
        """
        pre_res_dict = YOLOV2Tool.split_predict(
            out,
            self.anchor_number,
        )
        pre_txtytwth = pre_res_dict.get('position')[0]  # (N, H, W, a_n, 4)
        pre_xyxy = self.txtytwth_xyxy(pre_txtytwth)  # scaled on image

        pre_conf = torch.sigmoid(pre_res_dict.get('conf'))  # (N, H, W, a_n)
        pre_cls_prob = torch.softmax(pre_res_dict.get('cls_prob'), dim=-1)  # (N, H, W, a_n, kinds_number)
        pre_txty_s_twth = torch.cat(
            (torch.sigmoid(pre_txtytwth[..., 0:2]), pre_txtytwth[..., 2:4]),
            dim=-1
        )

        """
        split target
        """
        gt_res_dict = YOLOV2Tool.split_target(
            gt,
            self.anchor_number,
        )
        gt_xyxy = gt_res_dict.get('position')[1]  # (N, H, W, a_n, 4) scaled on image
        gt_txty_s_twth = self.xyxy_txty_s_twth(gt_xyxy)
        gt_conf_and_weight = gt_res_dict.get('conf')  # (N, H, W, a_n)
        # gt_conf = (gt_conf_and_weight > 0).float()
        gt_weight = gt_conf_and_weight
        gt_cls_prob = gt_res_dict.get('cls_prob')

        # get mask
        positive = (gt_weight > 0).float()
        ignore = (gt_weight == -1.0).float()
        negative = 1.0 - positive - ignore

        # compute loss
        # position loss
        temp = self.mse(pre_txty_s_twth, gt_txty_s_twth).sum(dim=-1)
        position_loss = torch.sum(
            temp * positive * gt_weight
        ) / N

        # conf loss
        # compute iou
        iou = YOLOV2Tool.iou(pre_xyxy, gt_xyxy)
        iou = iou.detach()  # (N, H, W, a_n) and no grad!

        # has obj/positive loss
        temp = self.mse(pre_conf, iou)
        has_obj_loss = torch.sum(
            temp * positive
        ) / N

        # no obj/negative loss
        temp = self.mse(pre_conf, torch.zeros_like(pre_conf).to(pre_conf.device))
        no_obj_loss = torch.sum(
            temp * negative
        ) / N

        # cls prob loss
        temp = self.mse(pre_cls_prob, gt_cls_prob).sum(dim=-1)
        cls_prob_loss = torch.sum(
            temp * positive
        ) / N

        # total loss
        total_loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_loss + \
            self.weight_conf_no_obj * no_obj_loss + \
            self.weight_cls_prob * cls_prob_loss

        loss_dict = {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'has_obj_loss': has_obj_loss,
            'no_obj_loss': no_obj_loss,
            'cls_prob_loss': cls_prob_loss
        }
        return loss_dict
