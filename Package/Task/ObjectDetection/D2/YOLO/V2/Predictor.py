import torch
from typing import List
from .Tools import YOLOV2Tool
from Package.Task.ObjectDetection.D2.Dev import DevPredictor


class YOLOV2Predictor(DevPredictor):

    def __init__(
            self,
            iou_th: float,
            prob_th: float,
            conf_th: float,
            score_th: float,
            pre_anchor_w_h_rate: tuple,
            kinds_name: list,
            image_size: tuple,
            image_shrink_rate: tuple
    ):
        super().__init__(
            iou_th,
            prob_th,
            conf_th,
            score_th,
            pre_anchor_w_h_rate,
            kinds_name,
            image_size,
            image_shrink_rate
        )

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

    def decode_one_target(
            self,
            x: torch.Tensor,
    ) -> List:

        a_n = len(self.pre_anchor_w_h)
        #####################################################################

        res_dict = YOLOV2Tool.split_target(
            x,
            a_n
        )

        conf = (res_dict.get('conf') > 0).float()
        cls_prob = res_dict.get('cls_prob')
        position_abs = res_dict.get('position')[1]  # scaled on image

        #####################################################################

        position_abs_ = position_abs.contiguous().view(-1, 4)
        conf_ = conf.contiguous().view(-1, )  # type: torch.Tensor
        cls_prob_ = cls_prob.contiguous().view(-1, len(self.kinds_name))
        scores_ = cls_prob_ * conf_.unsqueeze(-1).expand_as(cls_prob_)
        # (-1, kinds_num)

        cls_prob_mask = cls_prob_.max(dim=-1)[0] > self.prob_th  # type: torch.Tensor
        # (-1, )

        conf_mask = conf_ > self.conf_th  # type: torch.Tensor
        # (-1, )

        scores_max_value, scores_max_index = scores_.max(dim=-1)
        # (-1, )

        scores_mask = scores_max_value > self.score_th  # type: torch.Tensor
        # (-1, )

        mask = (conf_mask.float() * cls_prob_mask.float() * scores_mask.float()).bool()
        # (-1, )

        return self.nms(
            position_abs_[mask],
            scores_max_value[mask],
            scores_max_index[mask]
        )

    def decode_target(
            self,
            x: torch.Tensor,
    ) -> List[List]:
        res = []
        for i in range(x.shape[0]):
            pre_ = self.decode_one_target(
                x[i].unsqueeze(dim=0),
            )
            res.append(pre_)

        return res

    def decode_one_predict(
            self,
            out_put: torch.Tensor,
    ) -> List:
        a_n = len(self.pre_anchor_w_h)

        ##############################################

        res_dict = YOLOV2Tool.split_predict(
            out_put,
            a_n
        )
        conf = torch.sigmoid(res_dict.get('conf'))
        cls_prob = torch.softmax(res_dict.get('cls_prob'), dim=-1)
        position = res_dict.get('position')[0]
        position_abs = YOLOV2Tool.xywh_to_xyxy(
            position,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number
        ).clamp_(0, self.image_size[0] - 1)
        # scaled on image
        ###################################################

        position_abs_ = position_abs.contiguous().view(-1, 4)
        conf_ = conf.contiguous().view(-1, )  # type: torch.Tensor
        cls_prob_ = cls_prob.contiguous().view(-1, len(self.kinds_name))
        scores_ = cls_prob_ * conf_.unsqueeze(-1).expand_as(cls_prob_)
        # (-1, kinds_num)

        cls_prob_mask = cls_prob_.max(dim=-1)[0] > self.prob_th  # type: torch.Tensor
        # (-1, )

        conf_mask = conf_ > self.conf_th  # type: torch.Tensor
        # (-1, )

        scores_max_value, scores_max_index = scores_.max(dim=-1)
        # (-1, )

        scores_mask = scores_max_value > self.score_th  # type: torch.Tensor
        # (-1, )

        mask = (conf_mask.float() * cls_prob_mask.float() * scores_mask.float()).bool()
        # (-1, )

        return self.nms(
            position_abs_[mask],
            scores_max_value[mask],
            scores_max_index[mask]
        )

    def decode_predict(
            self,
            x: torch.Tensor,
    ) -> List[List]:
        res = []
        for i in range(x.shape[0]):
            pre_ = self.decode_one_predict(
                x[i].unsqueeze(dim=0),
            )
            res.append(pre_)

        return res

