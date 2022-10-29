from Package.Task.FacialRecognition.D2.Dev import DevModel, DevTool
from .mobile_face_net import MobileFaceNet
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
            self,
            feature_num: int,
            class_num: int,
            margin: float = 0.5,
            scale: float = 64.0
    ):
        super().__init__()
        self.feature_num = feature_num
        self.class_num = class_num
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(
            torch.Tensor(class_num, feature_num)
        )
        # torch linear weight !!!
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(
            self,
            feature: torch.Tensor,
            target: torch.Tensor = None
    ):
        """
        normal situation:
            out = feature @ weight.t()
                = ||feature|| * ||weight.t()|| * cos_theta
            (batch_num, kind_num)
        """
        feature_norm = DevTool.l2_norm(feature, dim=1)
        weight_t_norm = DevTool.l2_norm(self.weight.t(), dim=0)
        """
        ||feature_norm|| or  ||weight_t_norm||
        value is one ...
        """
        cos_theta = feature_norm @ weight_t_norm  # batch_num * kind_num
        theta = torch.arccos(cos_theta)
        cos_theta_and_m = torch.cos(theta + self.margin)
        one_hot = F.one_hot(target, num_classes=self.class_num).float().to(target.device)
        out = one_hot * cos_theta_and_m + (1.0 - one_hot) * cos_theta
        out = self.scale * out
        return out


class ArcFaceModel(DevModel):
    def __init__(
            self,
            net: Union[MobileFaceNet],
            feature_num: int,
            class_num: int,
            margin: float = 0.5,
            scale: float = 64.0
    ):
        super().__init__(net)
        # self.classifier_head = nn.Linear(feature_num, class_num)

        self.arc_face_head = ArcFaceHead(
            feature_num,
            class_num,
            margin,
            scale
        )

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ):
        feature = self.net(x)
        if self.training:
            out = self.arc_face_head(feature, kwargs.get('target'))
        else:
            out = None
        # out = self.classifier_head(feature)
        return {
                "feature": feature,
                "out": out,
            }
