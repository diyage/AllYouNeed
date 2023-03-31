from Package.Task.ObjectDetection.D2.Dev.model import DevModel
import torch
import torch.nn as nn
from typing import Union


class Focus(nn.Module):
    def __init__(
            self,
            in_channel: int = 3,
            out_channel: int = 32,
    ):
        super().__init__()

        self.out_map = nn.Sequential(
            CBA(in_channel * 4, out_channel, 3, 1, 1, nn.LeakyReLU)
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


class CBA(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel_size: int,
            stride: int,
            padding: int,
            act_type: Union[nn.LeakyReLU, nn.SiLU, nn.Mish] = nn.LeakyReLU
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
            act_type(),

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
            CBA(channel, channel // 2, 1, 1, 0, act_type=nn.LeakyReLU),
            CBA(channel // 2, channel, 3, 1, 1, act_type=nn.LeakyReLU),
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        origin = x
        part_out = self.out_map(x)
        return origin + part_out


class CSP1X(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            x: int
    ):
        super().__init__()
        mid_channel = out_channel // 2

        self.one = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, 1, 0)
        )
        self.two = nn.Sequential(
            CBA(in_channel, mid_channel, 1, 1, 0, act_type=nn.LeakyReLU),
            *[ResUnit(mid_channel) for _ in range(x)],
            nn.Conv2d(mid_channel, mid_channel, 1, 1, 0)
        )
        self.out_map = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            CBA(out_channel, in_channel, 1, 1, 0, act_type=nn.LeakyReLU)
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
        act_type
) -> nn.Sequential:
    return nn.Sequential(
        CBA(in_channel, in_channel, 1, 1, 0, act_type),
        CBA(in_channel, in_channel, 3, 1, 1, act_type),
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

        self.one = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, 1, 0)
        )
        self.two = nn.Sequential(
            CBA(in_channel, mid_channel, 1, 1, 0, act_type=nn.LeakyReLU),
            *[
                make_double_cba(mid_channel, act_type=nn.LeakyReLU) for _ in range(x)
            ],
            nn.Conv2d(mid_channel, mid_channel, 1, 1, 0)
        )
        self.out_map = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            CBA(out_channel, out_channel, 1, 1, 0, act_type=nn.LeakyReLU)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        part_one = self.one(x)
        part_two = self.two(x)
        o = torch.cat((part_one, part_two), dim=1)
        return self.out_map(o)


class SPP(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            pool_size: tuple = (5, 9, 13),
    ):
        super().__init__()
        mid_channel = out_channel // 2
        self.input_map = CBA(in_channel, mid_channel, 1, 1, 0, nn.LeakyReLU)

        self.pool_list = [
            nn.MaxPool2d(size, stride=1, padding=size // 2) for size in pool_size
        ]

        self.out_map = CBA(mid_channel * (len(pool_size) + 1), out_channel, 1, 1, 0, nn.LeakyReLU)

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


class YOLOV5SBackBone(nn.Module):
    def __init__(
            self,
            in_channel,
            base_channel: int = 32
    ):
        super().__init__()
        self.focus = Focus(in_channel, base_channel)  # 32

        self.cbl_0 = CBA(base_channel, base_channel * (2 ** 1), 3, 2, 1, nn.LeakyReLU)  # 64
        self.csp1_1_a = CSP1X(base_channel * 2, base_channel * 2, 1)  # CSP1 repeat x=1, named a

        self.cbl_1 = CBA(base_channel * (2 ** 1), base_channel * (2 ** 2), 3, 2, 1, nn.LeakyReLU)  # 128
        self.csp1_3_b = CSP1X(base_channel * (2 ** 2), base_channel * (2 ** 2), 3)  # CSP1 repeat x=3, named b

        self.cbl_2 = CBA(base_channel * (2 ** 2), base_channel * (2 ** 3), 3, 2, 1, nn.LeakyReLU)  # 256
        self.csp1_3_c = CSP1X(base_channel * (2 ** 3), base_channel * (2 ** 3), 3)  # CSP1 repeat x=3, named c

        self.cbl_3 = CBA(base_channel * (2 ** 3), base_channel * (2 ** 4), 3, 2, 1, nn.LeakyReLU)  # 512

        self.spp = SPP(base_channel * (2 ** 4), base_channel * (2 ** 4))

    def forward(
            self,
            x: torch.Tensor
    ):
        x = self.csp1_3_b(self.cbl_1(self.csp1_1_a(self.cbl_0(self.focus(x)))))
        out_of_csp_1_3_b = x  # 128
        x = self.csp1_3_c(self.cbl_2(x))
        out_of_csp_1_3_c = x  # 256
        x = self.spp(self.cbl_3(x))
        out_of_spp = x  # 512
        return out_of_csp_1_3_b, out_of_csp_1_3_c, out_of_spp


class YOLOV5SNeck(nn.Module):
    def __init__(
            self,
            max_in_channel,
    ):
        super().__init__()
        in_channel = max_in_channel
        self.csp_2_1_a = CSP2X(in_channel, in_channel, 1)
        # 512 --> 512
        self.cbl_0 = CBA(in_channel, in_channel // 2, 1, 1, 0, nn.LeakyReLU)
        # 512 --> 256
        self.up = nn.Upsample(scale_factor=2.0)

        self.csp_2_1_b = CSP2X(in_channel, in_channel // (2 ** 1), 1)  # input element is (two tensors cat)
        # 512 --> 256
        self.cbl_1 = CBA(in_channel // (2 ** 1), in_channel // (2 ** 2), 1, 1, 0, nn.LeakyReLU)
        # 256 --> 128
        self.csp_2_1_c = CSP2X(in_channel // (2 ** 1), in_channel // (2 ** 2), 1)  # input element is (two tensors cat)
        # 256 --> 128

        self.cbl_2 = CBA(in_channel // (2 ** 2), in_channel // (2 ** 2), 3, 2, 1, nn.LeakyReLU)
        # 128 --> 128

        self.csp_2_1_d = CSP2X(in_channel // (2 ** 1), in_channel // (2 ** 1), 1)  # input element is (two tensors cat)
        # 256 --> 256

        self.cbl_3 = CBA(in_channel // (2 ** 1), in_channel // (2 ** 1), 3, 2, 1, nn.LeakyReLU)
        # 256 --> 256
        self.csp_2_1_e = CSP2X(in_channel, in_channel, 1)

    def forward(
            self,
            out_of_csp_1_3_b: torch.Tensor,
            out_of_csp_1_3_c: torch.Tensor,
            out_of_spp: torch.Tensor
    ):
        temp_a = self.cbl_0(self.csp_2_1_a(out_of_spp))

        temp_b = self.cbl_1(self.csp_2_1_b(
            torch.cat(
                (
                    self.up(temp_a),
                    out_of_csp_1_3_c
                ), dim=1
            ),

        ))

        out_of_csp_2_1_c = self.csp_2_1_c(torch.cat(
            (
                self.up(temp_b),
                out_of_csp_1_3_b
            ),
            dim=1
        ))
        out_of_csp_2_1_d = self.csp_2_1_d(torch.cat(
            (
                self.cbl_2(out_of_csp_2_1_c),
                temp_b
            ),
            dim=1
        ))

        out_of_csp_2_1_e = self.csp_2_1_e(torch.cat(
            (
                self.cbl_3(out_of_csp_2_1_d),
                temp_a
            ),
            dim=1
        ))
        return out_of_csp_2_1_c, out_of_csp_2_1_d, out_of_csp_2_1_e


class YOLOV5SPrediction(nn.Module):
    def __init__(
            self,
            single_anchor_num: int = 3,
            channel_list: list = [128, 256, 512],
            kinds_num: int = 80
    ):
        super().__init__()
        self.head_s = nn.Conv2d(channel_list[0], single_anchor_num*(kinds_num + 5), 1, 1, 0)
        self.head_m = nn.Conv2d(channel_list[1], single_anchor_num*(kinds_num + 5), 1, 1, 0)
        self.head_l = nn.Conv2d(channel_list[2], single_anchor_num*(kinds_num + 5), 1, 1, 0)

    def forward(
            self,
            out_of_csp_2_1_c: torch.Tensor,
            out_of_csp_2_1_d: torch.Tensor,
            out_of_csp_2_1_e: torch.Tensor
    ):
        out_for_s = self.head_s(out_of_csp_2_1_c)
        out_for_m = self.head_m(out_of_csp_2_1_d)
        out_for_l = self.head_l(out_of_csp_2_1_e)
        return {
            'for_s': out_for_s,
            'for_m': out_for_m,
            'for_l': out_for_l,
        }


class YOLOV5Model(nn.Module):
    def __init__(
            self,
            in_channel: int = 3,
            base_channel: int = 32,
            kinds_num: int = 80,
            single_anchor_num: int = 3
    ):
        super().__init__()
        self.backbone = YOLOV5SBackBone(
            in_channel,
            base_channel
        )
        self.neck = YOLOV5SNeck(
            max_in_channel=base_channel*(2**4)
        )
        self.prediction = YOLOV5SPrediction(
            single_anchor_num=single_anchor_num,
            channel_list=[base_channel*(2**2), base_channel*(2**3), base_channel*(2**4)],
            kinds_num=kinds_num,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> dict:
        a, b, c = self.backbone(x)
        a, b, c = self.neck(a, b, c)
        res = self.prediction(a, b, c)
        return res


"""
debug something
"""


def debug_focus():
    img = torch.rand(size=(8, 3, 608, 608))
    m = Focus(3, 32)
    y = m(img)
    print(y.shape)


def debug_cba():
    img = torch.rand(size=(8, 3, 608, 608))
    m = CBA(3, 32, 3, 2, 1)
    y = m(img)
    print(y.shape)


def debug_res_unit():
    img = torch.rand(size=(8, 32, 608, 608))
    m = ResUnit(32)
    y = m(img)
    print(y.shape)


def debug_csp1x():
    img = torch.rand(size=(8, 64, 608, 608))
    m = CSP1X(64, 64, x=3)
    y = m(img)
    print(y.shape)


def debug_csp2x():
    img = torch.rand(size=(8, 64, 608, 608))
    m = CSP2X(64, 64, x=3)
    y = m(img)
    print(y.shape)


def debug_spp():
    img = torch.rand(size=(8, 64, 608, 608))
    m = SPP(64, 128)
    y = m(img)
    print(y.shape)


def debug_backbone():
    img = torch.rand(size=(8, 3, 608, 608))
    m = YOLOV5SBackBone(3, base_channel=32)
    a, b, c = m(img)
    print(a.shape)
    print(b.shape)
    print(c.shape)


def debug_neck():
    img = torch.rand(size=(8, 3, 608, 608))
    m = YOLOV5SBackBone(3, base_channel=32)
    a, b, c = m(img)
    n = YOLOV5SNeck(512)
    a, b, c = n(a, b, c)
    print(a.shape)
    print(b.shape)
    print(c.shape)


def debug_prediction():
    img = torch.rand(size=(8, 3, 608, 608))
    m = YOLOV5SBackBone(3, base_channel=32)
    a, b, c = m(img)
    n = YOLOV5SNeck(512)
    a, b, c = n(a, b, c)
    p = YOLOV5SPrediction(kinds_num=80)
    res = p(a, b, c)
    for key, val in res.items():
        print("{}: {}".format(key, val.shape))


def debug_model():
    img = torch.rand(size=(8, 3, 608, 608))
    m = YOLOV5Model(3, base_channel=32, kinds_num=80, single_anchor_num=3)
    res = m(img)

    for key, val in res.items():
        print("{}: {}".format(key, val.shape))


def compute_model_size():

    total = 0
    m = YOLOV5Model(3, base_channel=32, kinds_num=80, single_anchor_num=3)
    for para in m.parameters():
        if para.requires_grad:
            total += para.numel()
    print('Total(could be trained): {:.1f}M'.format(1.0*total/1e+6))


if __name__ == '__main__':
    print('DEBUG:')
    # debug_focus()
    # debug_cba()
    # debug_res_unit()
    # debug_csp1x()
    # debug_csp2x()
    # debug_spp()
    # debug_backbone()
    # debug_neck()
    # debug_prediction()
    # debug_model()
    compute_model_size()
