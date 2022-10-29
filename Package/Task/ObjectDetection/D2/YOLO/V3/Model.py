import torch
import torch.nn as nn
import torch.nn.functional as F
from Package.Task.ObjectDetection.D2.Dev import DevModel
from typing import Union


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1):
        '''

        Args:
            c1: in_channel
            c2: out_channel
            k:
            p:
            s:
            d:
        '''
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, padding=p, stride=s, dilation=d),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                # two CBLs
                Conv_BN_LeakyReLU(ch, ch//2, k=1),
                Conv_BN_LeakyReLU(ch//2, ch, k=3, p=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x   # in channel == out channel


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """

    def __init__(
            self,
            num_classes=1000,
    ):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, k=3, p=1),
            Conv_BN_LeakyReLU(32, 64, k=3, p=1, s=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, k=3, p=1, s=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, k=3, p=1, s=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, k=3, p=1, s=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, k=3, p=1, s=2),
            resblock(1024, nblocks=4)
        )

        """
        # used for classification
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, num_classes)
        we just used pre-trained model,
        so this part could be ignored...
        """

    def forward(self, x: torch.Tensor):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        # x = self.avgpool(c5)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return c3, c4, c5


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


def get_backbone_darknet_53(
        path: str = None
) -> DarkNet_53:
    model = DarkNet_53()
    if path is not None:
        model.load_state_dict(torch.load(path), strict=False)
        print('Load pre-trained darknet53 successfully!'.center(50, '*'))
    return model


class YOLOV3Model(DevModel):
    def __init__(
            self,
            net: Union[nn.Module, DarkNet_53],
            num_anchors_for_single_size: int = 3,
            num_classes: int = 20,
    ):
        super().__init__(net)
        self.num_classes = num_classes
        self.num_anchors = num_anchors_for_single_size
        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        self.conv_1x1_3 = Conv(512, 256, k=1)
        self.extra_conv_3 = Conv(512, 1024, k=3, p=1)
        self.pred_3 = nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)
        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)
        self.extra_conv_2 = Conv(256, 512, k=3, p=1)
        self.pred_2 = nn.Conv2d(512, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)
        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )
        self.extra_conv_1 = Conv(128, 256, k=3, p=1)
        self.pred_1 = nn.Conv2d(256, self.num_anchors * (1 + 4 + self.num_classes), kernel_size=1)

    def forward(
            self,
            x: torch.Tensor
    ):
        # backbone
        c3, c4, c5 = self.net(x)

        # FPN, 多尺度特征融合
        p5 = self.conv_set_3(c5)
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.conv_set_2(p4)
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.conv_set_1(p3)

        # head
        # s = 32, 预测大物体
        p5 = self.extra_conv_3(p5)
        pred_3 = self.pred_3(p5)

        # s = 16, 预测中物体
        p4 = self.extra_conv_2(p4)
        pred_2 = self.pred_2(p4)

        # s = 8, 预测小物体
        p3 = self.extra_conv_1(p3)
        pred_1 = self.pred_1(p3)

        res = {
            'for_s': pred_1,
            'for_m': pred_2,
            'for_l': pred_3,
        }
        return res

