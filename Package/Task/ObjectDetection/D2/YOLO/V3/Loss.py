import torch
import torch.nn as nn
from Package.Task.ObjectDetection.D2.Dev import DevLoss
from .Tools import YOLOV3Tool


class YOLOV3Loss(DevLoss):
    def __init__(
            self,
            pre_anchor_w_h_rate: dict,
            image_shrink_rate: dict,
            each_size_anchor_number: int = 3,
            weight_position: float = 1.0,
            weight_conf_has_obj: float = 1.0,
            weight_conf_no_obj: float = 1.0,
            weight_cls_prob: float = 1.0,
            image_size: tuple = (416, 416),
    ):
        super().__init__()
        self.pre_anchor_w_h_rate = pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = image_shrink_rate
        self.grid_number = None
        self.image_size = None

        self.change_image_wh(image_size)

        self.each_size_anchor_number = each_size_anchor_number
        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_cls_prob = weight_cls_prob

        self.anchor_keys = list(pre_anchor_w_h_rate.keys())
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce_l = nn.BCEWithLogitsLoss(reduction='none')

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV3Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def forward(
            self,
            out_put: dict,
            target: dict,
    ):
        res_out = YOLOV3Tool.split_predict(
            out_put,
            self.each_size_anchor_number
        )
        res_target = YOLOV3Tool.split_target(
            target,
            self.each_size_anchor_number
        )
        loss_dict = {
            'total_loss': 0.0,
            'position_loss': 0.0,
            'has_obj_loss': 0.0,
            'no_obj_loss': 0.0,
            'cls_prob_loss': 0.0
        }

        for anchor_key in self.anchor_keys:
            pre_res_dict = res_out[anchor_key]
            gt_res_dict = res_target[anchor_key]

            N = out_put[anchor_key].shape[0]
            # -------------------------------------------------------------------
            # split output
            pre_txtytwth = pre_res_dict.get('position')[0]  # (N, H, W, a_n, 4)

            pre_txty = pre_txtytwth[..., 0:2]  # (N, H, W, a_n, 2)
            # be careful, not use sigmoid on pre_txty
            pre_twth = pre_txtytwth[..., 2:4]  # (N, H, W, a_n, 2)

            pre_xyxy = YOLOV3Tool.xywh_to_xyxy(
                pre_txtytwth,
                self.pre_anchor_w_h[anchor_key],
                self.grid_number[anchor_key]
            )
            # scaled in [0, 1]

            pre_conf = torch.sigmoid(pre_res_dict.get('conf'))  # (N, H, W, a_n)

            pre_cls_prob = pre_res_dict.get('cls_prob')  # (N, H, W, a_n, kinds_num)
            # be careful, if you use mse --> please softmax(pre_cls_prob)
            # otherwise not (softmax already used in CrossEntropy of PyTorch)
            # pre_cls_prob = torch.softmax(pre_res_dict.get('cls_prob'), dim=-1)  # (N, H, W, a_n, kinds_number)

            # -------------------------------------------------------------------
            # split target

            gt_xyxy = gt_res_dict.get('position')[1]
            # (N, H, W, a_n, 4) scaled in [0, 1]

            gt_txty_s_twth = YOLOV3Tool.xyxy_to_xy_s_wh(
                gt_xyxy,
                self.pre_anchor_w_h[anchor_key],
                self.grid_number[anchor_key]
            )
            gt_txty_s = gt_txty_s_twth[..., 0:2]  # (N, H, W, a_n, 2)
            gt_twth = gt_txty_s_twth[..., 2:4]  # (N, H, W, a_n, 2)

            gt_conf_and_weight = gt_res_dict.get('conf')  # (N, H, W, a_n)
            # gt_conf = (gt_conf_and_weight > 0).float()
            gt_weight = gt_conf_and_weight

            gt_cls_prob = gt_res_dict.get('cls_prob').argmax(dim=-1)  # (N, H, W, a_n)
            # be careful, if you use CrossEntropy of PyTorch, please argmax gt_res_dict.get('cls_prob')
            # because gt_res_dict.get('cls_prob') is one-hot code
            # gt_cls_prob = gt_res_dict.get('cls_prob')

            # -------------------------------------------------------------------
            # compute mask
            positive = (gt_weight > 0).float()
            ignore = (gt_weight == -1.0).float()
            negative = 1.0 - positive - ignore

            # -------------------------------------------------------------------
            # compute loss

            # position loss
            temp = (self.bce_l(pre_txty, gt_txty_s) + self.mse(pre_twth, gt_twth)).sum(dim=-1)
            loss_dict['position_loss'] += torch.sum(
                temp * positive * gt_weight
            ) / N

            # conf loss
            # compute iou
            iou = YOLOV3Tool.iou(pre_xyxy, gt_xyxy)
            iou = iou.detach()  # (N, H, W, a_n) and no grad!

            # has obj/positive loss
            temp = self.mse(pre_conf, iou)
            loss_dict['has_obj_loss'] += torch.sum(
                temp * positive
            ) / N

            # no obj/negative loss
            temp = self.mse(pre_conf, torch.zeros_like(pre_conf).to(pre_conf.device))
            loss_dict['no_obj_loss'] += torch.sum(
                temp * negative
            ) / N

            # cls prob loss
            temp = self.ce(
                pre_cls_prob.contiguous().view(-1, pre_cls_prob.shape[-1]),
                gt_cls_prob.contiguous().view(-1,)
            )
            # temp = self.mse(pre_cls_prob, gt_cls_prob).sum(dim=-1)
            loss_dict['cls_prob_loss'] += torch.sum(
                temp * positive.contiguous().view(-1,)
            ) / N

        loss_dict['total_loss'] = self.weight_position * loss_dict['position_loss'] + \
            self.weight_conf_has_obj * loss_dict['has_obj_loss'] + \
            self.weight_conf_no_obj * loss_dict['no_obj_loss'] + \
            self.weight_cls_prob * loss_dict['cls_prob_loss']

        return loss_dict
