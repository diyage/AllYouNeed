import torch
import torch.nn as nn
import torch.nn.functional as F
from Package.Task.ObjectDetection.D2.YOLO.V3.Model import get_backbone_darknet_53, Conv
from .model_original import YOLOVXDecoupledHead


class YOLOVXBackBoneBasedDarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = get_backbone_darknet_53('/home/dell/PycharmProjects/YOLO/pre_trained/darknet53_75.42.pth')

    def forward(
            self,
            x: torch.Tensor
    ):
        c3, c4, c5 = self.net(x)
        return c3, c4, c5


class YOLOVXNeckBasedDarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        self.conv_1x1_3 = Conv(512, 256, k=1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )

    def forward(
            self,
            c3: torch.Tensor,
            c4: torch.Tensor,
            c5: torch.Tensor,
    ):
        # FPN, 多尺度特征融合
        p5 = self.conv_set_3(c5)
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.conv_set_2(p4)
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.conv_set_1(p3)
        return p3, p4, p5


def get_back_bone_based_dark_net_53(
) -> YOLOVXBackBoneBasedDarkNet53:
    return YOLOVXBackBoneBasedDarkNet53(
    )


def get_neck_based_dark_net_53(
) -> YOLOVXNeckBasedDarkNet53:

    return YOLOVXNeckBasedDarkNet53(
    )


def get_head_based_dark_net_53(
        wide_mul,
        cls_num
) -> nn.ModuleList:
    feature_channel = (
        128,
        256,
        512
    )
    return nn.ModuleList(
        YOLOVXDecoupledHead(
            channel,
            wide_mul,
            cls_num
        ) for channel in feature_channel
    )
