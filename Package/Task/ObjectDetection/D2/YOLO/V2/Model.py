from Package.Task.ObjectDetection.D2.Dev import DevModel
import torch
import torch.nn as nn
from typing import Union


def make_conv_bn_active_layer(
        in_channel: int,
        out_channel: int,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        active: nn.Module = nn.LeakyReLU):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel),
        active(),
    )


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self,  num_classes=1000):
        # https://zhuanlan.zhihu.com/p/105278156
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2 ,2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2 ,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 1024
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        self.conv_7 = nn.Conv2d(1024, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_extractor_top = None
        self.feature_extractor_down = None
        self.feature_extractor = None
        self.classifier = None
        self.translate_to_my_net()

    def translate_to_my_net(
            self
    ):
        tmp = [* self.conv_5]
        self.feature_extractor_top = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            *tmp[:-1],
        )
        self.feature_extractor_down = nn.Sequential(
            tmp[-1],
            self.conv_6,
        )
        self.feature_extractor = nn.Sequential(
            self.feature_extractor_top,
            self.feature_extractor_down,
        )
        self.classifier = nn.Sequential(
            self.conv_7,
            self.avgpool
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        x = self.conv_7(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def get_backbone_dark_net_19(
        path: str = None
) -> DarkNet_19:
    m = DarkNet_19()
    if path is not None:
        print('init pre-trained dark net 19'.center(50, '*'))
        saved_state_dict = torch.load(path)
        m.load_state_dict(saved_state_dict, strict=False)
        print('init successfully!'.center(50, '*'))
    return m


class YOLOV2Model(DevModel):
    def __init__(
            self,
            net: Union[nn.Module, DarkNet_19],
            num_anchors_for_single_size: int = 5,
            num_classes: int = 20,
    ):
        super().__init__(net)
        self.num_classes = num_classes
        self.num_anchors = num_anchors_for_single_size
        """
        yolo v2 just has one scaled size (13*13)
        and anchor number is 5  
        """
        self.pass_through_conv = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=(1, 1))
        )

        self.conv3_1024 = nn.Sequential(
            make_conv_bn_active_layer(1024, 1024),
            make_conv_bn_active_layer(1024, 1024),
            make_conv_bn_active_layer(1024, 1024),
        )

        self.conv3_1_out = nn.Sequential(
            make_conv_bn_active_layer(1280, 1024),
            nn.Conv2d(1024, self.num_anchors*(self.num_classes + 5), kernel_size=(1, 1))
        )

    def pass_through(
            self,
            a: torch.Tensor,
            stride: tuple = (2, 2)
    ):
        # https://zhuanlan.zhihu.com/p/35325884
        N = a.shape[0]
        assert a.shape == (N, 512, 26, 26)

        x = self.pass_through_conv(a)  # N * 64 * 26 * 26
        # ReOrg
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        ws, hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)

        return x  # N * 256 * 13 * 13

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        assert x.shape == (N, 3, 416, 416)

        a = self.net.feature_extractor_top(x)  # N * 512 * 26 * 26
        a_ = self.pass_through(a)  # N * 256 * 13 * 13

        b = self.net.feature_extractor_down(a)  # N * 1024 * 13 * 13
        b_ = self.conv3_1024(b)  # N * 1024 * 13 * 13

        d = torch.cat((b_, a_), dim=1)  # N * (1024 + 256) * 13 * 13

        return self.conv3_1_out(d)  # N * 125 * 13 * 13
