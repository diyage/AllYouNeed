#
# 这里定义的是YOLO VX 模型结构，参考的源代码地址（原文放出的地址）为：https://github.com/Megvii-BaseDetection/YOLOX
# 许多代码选择将训练阶段的损失函数计算，以及推理阶段的box解码都放在model里，我不是特别喜欢

# YOLOVXModel 继承至 DevModel，其中 net 就是back_bone 代表主干网络

import torch
import torch.nn as nn
from typing import *


class CBS(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel_size: int,
            stride: int,
            padding: int,
    ):
        super().__init__()
        self.out_map = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                (kernel_size, kernel_size),
                (stride, stride),
                (padding, padding)
            ),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(),
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        o = self.out_map(x)
        return o


class ResUnit(nn.Module):
    def __init__(
            self,
            channel: int
    ):
        super().__init__()
        self.out_map = nn.Sequential(
            CBS(channel, channel // 2, 1, 1, 0),
            CBS(channel // 2, channel, 3, 1, 1),
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        origin = x
        part_out = self.out_map(x)
        return origin + part_out


class SPP(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            pool_size: tuple = (5, 9, 13),
    ):
        super().__init__()
        mid_channel = out_channel // 2
        self.input_map = CBS(in_channel, mid_channel, 1, 1, 0)

        self.pool_list = nn.ModuleList([
            nn.MaxPool2d(size, stride=1, padding=size // 2) for size in pool_size
        ])

        self.out_map = CBS(mid_channel * (len(pool_size) + 1), out_channel, 1, 1, 0)

    def forward(
            self,
            x: torch.Tensor
    ):
        x = self.input_map(x)
        pool_feature = [m(x) for m in self.pool_list]
        feature_list = [x] + pool_feature
        o = torch.cat(
            feature_list,
            dim=1
        )
        return self.out_map(o)


class Focus(nn.Module):
    def __init__(
            self,
            in_channel: int = 3,
            out_channel: int = 32,
    ):
        super().__init__()

        self.out_map = nn.Sequential(
            CBS(in_channel * 4, out_channel, 3, 1, 1)
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        a = x[:, :, ::2, ::2]
        b = x[:, :, 1::2, ::2]
        c = x[:, :, ::2, 1::2]
        d = x[:, :, 1::2, 1::2]

        o = torch.cat([a, b, c, d], dim=1)
        o = self.out_map(o)
        return o


class CSP1X(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            x: int
    ):
        super().__init__()
        mid_channel = out_channel // 2

        self.one = CBS(in_channel, mid_channel, 1, 1, 0)

        self.two = nn.Sequential(
            CBS(in_channel, mid_channel, 1, 1, 0),
            *[ResUnit(mid_channel) for _ in range(x)],
        )
        self.out_map = nn.Sequential(
            CBS(out_channel, in_channel, 1, 1, 0)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        part_one = self.one(x)
        part_two = self.two(x)
        o = torch.cat((part_one, part_two), dim=1)
        return self.out_map(o)


def make_double_cba(
        in_channel,
) -> nn.Sequential:
    return nn.Sequential(
        CBS(in_channel, in_channel, 1, 1, 0),
        CBS(in_channel, in_channel, 3, 1, 1),
    )


class CSP2X(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            x: int
    ):
        super().__init__()
        mid_channel = out_channel // 2

        self.one = CBS(in_channel, mid_channel, 1, 1, 0)

        self.two = nn.Sequential(
            CBS(in_channel, mid_channel, 1, 1, 0),
            *[
                make_double_cba(mid_channel) for _ in range(x)
            ],

        )
        self.out_map = nn.Sequential(
            CBS(out_channel, out_channel, 1, 1, 0)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        part_one = self.one(x)
        part_two = self.two(x)
        o = torch.cat((part_one, part_two), dim=1)
        return self.out_map(o)


class YOLOVXBackBoneOriginal(nn.Module):
    def __init__(
            self,
            in_channel: int = 3,
            base_channel: int = 32,
            base_deep: int = 3
    ):
        super().__init__()
        self.focus = Focus(
            in_channel,
            base_channel
        )
        self.dark2 = nn.Sequential(
            CBS(base_channel, base_channel * (2 ** 1), 3, 2, 1),
            CSP1X(base_channel * 2, base_channel * 2, 1*base_deep)
        )
        self.dark3 = nn.Sequential(
            CBS(base_channel * (2 ** 1), base_channel * (2 ** 2), 3, 2, 1),
            CSP1X(base_channel * (2 ** 2), base_channel * (2 ** 2), 3*base_deep)
        )

        self.dark4 = nn.Sequential(
            CBS(base_channel * (2 ** 2), base_channel * (2 ** 3), 3, 2, 1),
            CSP1X(base_channel * (2 ** 3), base_channel * (2 ** 3), 3*base_deep),
        )
        self.dark5 = nn.Sequential(
            CBS(base_channel * (2 ** 3), base_channel * (2 ** 4), 3, 2, 1),
            SPP(base_channel * (2 ** 4), base_channel * (2 ** 4)),
            CSP2X(base_channel * (2 ** 4), base_channel * (2 ** 4), 1*base_deep)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        c1 = self.focus(x)
        c2 = self.dark2(c1)
        c3 = self.dark3(c2)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)
        return c3, c4, c5


class YOLOVXNeckOriginal(nn.Module):
    def __init__(
            self,
            deep_mul: int,
            in_channel_vec: Sequence[int],
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2.0)
        # c/p5 --> c/p3
        self.a = nn.ModuleList([
            CBS(in_channel_vec[2], in_channel_vec[1], 1, 1, 0),
            CSP2X(2*in_channel_vec[1], in_channel_vec[1], 1*deep_mul),  # c3p4
            CBS(in_channel_vec[1], in_channel_vec[0], 1, 1, 0),
        ])
        # c/p3 --> c/p5
        self.b = nn.ModuleList([
            CSP2X(2*in_channel_vec[0], in_channel_vec[0], 1*deep_mul),  # c3p3
            CBS(in_channel_vec[0], in_channel_vec[0], 3, 2, 1),
            CSP2X(2*in_channel_vec[0], in_channel_vec[1], 1*deep_mul),  # c3n3
            CBS(in_channel_vec[1], in_channel_vec[1], 3, 2, 1),
            CSP2X(2*in_channel_vec[1], in_channel_vec[2], 1*deep_mul)
        ])

    def forward(
            self,
            c3: torch.Tensor,
            c4: torch.Tensor,
            c5: torch.Tensor,
    ):
        m5 = self.a[0](c5)
        temp = torch.cat(
            [c4, self.up(m5)],
            dim=1
        )
        m4 = self.a[2](self.a[1](temp))
        m3 = torch.cat(
            [c3, self.up(m4)],
            dim=1
        )

        p3 = self.b[0](m3)
        temp = torch.cat(
            [m4, self.b[1](p3)],
            dim=1
        )
        p4 = self.b[2](temp)

        temp = torch.cat(
            [m5, self.b[3](p4)],
            dim=1
        )
        p5 = self.b[4](temp)
        return p3, p4, p5


class YOLOVXDecoupledHead(nn.Module):
    def __init__(
            self,
            in_feature: int,
            wide_mul: int,
            cls_num: int = 80
    ):
        super().__init__()
        feature_mapping: int = 256 * wide_mul
        self.feature_map = CBS(in_feature, feature_mapping, 1, 1, 0)

        self.__block_1 = nn.Sequential(
            CBS(feature_mapping, feature_mapping, 3, 1, 1),
            CBS(feature_mapping, feature_mapping, 3, 1, 1),

            nn.Conv2d(feature_mapping, cls_num, 1, 1, 0),
            # nn.Sigmoid(),
            # 这里训练阶段是不需要sigmoid or softmax函数的，因为后续pytorch提供的损失函数集成的有，
            # 当然有时候我们更喜欢叫他 logits，这要求程序员自己要注意，在推理阶段，补上sigmoid等类似的激活操作
        )

        self.__block_2_1 = nn.Sequential(
            CBS(feature_mapping, feature_mapping, 3, 1, 1),
            CBS(feature_mapping, feature_mapping, 3, 1, 1),
        )
        self.__block_2_2_for_obj = nn.Conv2d(feature_mapping, 1, 1, 1, 0)
        self.__block_2_2_for_pos = nn.Conv2d(feature_mapping, 4, 1, 1, 0)

        self.__cls_num = cls_num

    def forward(
            self,
            x: torch.Tensor
    ):
        x = self.feature_map(x)

        o_for_cls = self.__block_1(x)
        o2_part = self.__block_2_1(x)
        o2_for_obj = self.__block_2_2_for_obj(o2_part)
        o2_for_pos = self.__block_2_2_for_pos(o2_part)

        o: torch.Tensor = torch.cat(
            (o2_for_pos, o2_for_obj, o_for_cls),
            dim=1
        )
        # shape --> [n, info_num, h, w]
        return o


def get_back_bone_original(
        in_channel,
        base_channel,
        wide_mul,
        deep_mul
) -> YOLOVXBackBoneOriginal:
    return YOLOVXBackBoneOriginal(
        in_channel,
        base_channel * wide_mul,
        deep_mul
    )


def get_neck_original(
        base_channel,
        wide_mul,
        deep_mul,
) -> YOLOVXNeckOriginal:

    feature_channel: Sequence[int] = (
        base_channel * wide_mul * 4,
        base_channel * wide_mul * 8,
        base_channel * wide_mul * 16
    )
    return YOLOVXNeckOriginal(
        deep_mul,
        in_channel_vec=feature_channel,
    )


def get_head_original(
        base_channel,
        wide_mul,
        cls_num,
) -> nn.ModuleList:
    feature_channel: Sequence[int] = (
        base_channel * wide_mul * 4,
        base_channel * wide_mul * 8,
        base_channel * wide_mul * 16
    )
    return nn.ModuleList(
        YOLOVXDecoupledHead(
            channel,
            wide_mul,
            cls_num
        ) for channel in feature_channel
    )
