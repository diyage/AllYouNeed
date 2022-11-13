from ..Dev import DevLoss
import torch
import torch.nn as nn


class MTCNNLoss(DevLoss):
    def __init__(
            self,
            cls_factor: float,
            box_factor: float,
            landmark_factor: float,
    ):
        super().__init__()
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    def forward(
            self,
            predict: dict,
            target: dict,
            train_net_type: str,
            image_type: int,
            *args,
            **kwargs
    ):
        assert train_net_type in ['p', 'r', 'o']
        assert image_type in [0, 1, 2, 3, 4, 5]
        # please see MTCNNDataSet
        # 0 used for computing key-point loss
        # 1 used for computing positive loss
        # 2, 3, 4 used for computing negative loss
        # 5 used for computing part loss

        out_key = '{}_out'.format(train_net_type)
        cls: torch.Tensor = predict[out_key]['cls']
        """
        for P-Net
            cls shape [batch, n, 1, 1]
        for R-Net/O-Net
            cls shape [batch, n]
        
        the same to pos_offset or key_point
        """
        pos_offset: torch.Tensor = predict[out_key]['pos_offset']
        key_point: torch.Tensor = predict[out_key]['key_point']

        pre_cls = cls.view(-1, 2)
        pre_pos_offset = pos_offset.view(-1, 4)
        pre_key_point_offset = key_point.view(-1, 10)

        gt_cls = target.get('cls')
        gt_pos_offset = target.get('pos_offset')  # used for positive or part images
        gt_key_point_offset = target.get('key_point')

        res = {}

        if image_type == 0:
            # 0 used for computing key-point loss
            loss = self.loss_landmark(pre_key_point_offset, gt_key_point_offset) * self.land_factor
            res['total_loss'] = loss
            res['key_point_loss'] = loss
            return res
        elif image_type == 1:
            # 1 used for computing positive loss
            # classification and box regression
            temp0 = self.loss_cls(pre_cls, gt_cls) * self.cls_factor
            temp1 = self.loss_box(
                pre_pos_offset,
                gt_pos_offset
            ) * self.box_factor

            res['total_loss'] = temp0 + temp1
            res['cls_loss'] = temp0
            res['position_loss'] = temp1
            return res
        elif image_type == 5:
            # 5 used for computing part loss
            # just used for box regression
            loss = self.loss_box(
                pre_pos_offset,
                gt_pos_offset
            ) * self.box_factor
            res['total_loss'] = loss
            res['position_loss'] = loss
            return res
        else:
            # 2, 3, 4 used for computing negative loss
            # just used for classification
            loss = self.loss_cls(pre_cls, gt_cls) * self.cls_factor
            res['total_loss'] = loss
            res['cls_loss'] = loss
            return res
