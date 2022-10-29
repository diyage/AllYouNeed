import torch
import torch.nn as nn
import torch.nn.functional as F
from Package.Task.ObjectDetection.D2.Dev import DevModel
from typing import Union


def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1):
    layers = []
    layers.append(nn.Conv2d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False))
    layers.append(nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03))
    layers.append(nn.Mish(inplace=True))
    return nn.Sequential(*layers)


def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       dilation=1):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                dilation=dilation
            )
        )
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 dilation=1,
                 downsample=None):
        super(DarkBlock, self).__init__()

        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-4, momentum=0.03)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03)

        self.conv1 = nn.Conv2d(
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity

        return out


class CrossStagePartialBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 dilation=1,
                 stride=2):
        super(CrossStagePartialBlock, self).__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.stage_layers = stage_layers

        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out


class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()

        self.block = DarkBlock
        self.stage_blocks = (1, 2, 8, 8, 4)
        self.with_csp = True
        self.inplanes = 32

        self.backbone = nn.ModuleDict()
        self.layer_names = []
        # First stem layer
        self.backbone["conv1"] = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.backbone["bn1"] = nn.BatchNorm2d(self.inplanes, eps=1e-4, momentum=0.03)
        self.backbone["act1"] = nn.Mish(inplace=True)

        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            layer = make_cspdark_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation
            )
            layer = CrossStagePartialBlock(
                self.inplanes,
                planes,
                stage_layers=layer,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation,
                stride=stride
            )
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.backbone[layer_name] = layer
            self.layer_names.append(layer_name)

    def forward(self, x):
        outputs = []
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["act1"](x)

        for i, layer_name in enumerate(self.layer_names):
            layer = self.backbone[layer_name]
            x = layer(x)
            outputs.append(x)
        return outputs[-3:]  # C3, C4, C5


def get_backbone_csp_darknet_53(
        path: str = None
):
    """
    Create a CSPDarkNet.
    """
    model = CSPDarkNet53()
    if path is not None:
        print('Loading the pretrained model ...')
        checkpoint = torch.load(path, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)
    return model


class CBL(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(
            self,
            x: torch.Tensor
    ):
        return self.act(self.bn(self.conv(x)))


class Process1(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        self.cbl_s_0 = nn.Sequential(
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 512, 1, 1, 0)
        )

        self.max_pool1 = nn.MaxPool2d(kernel_size=(13, 13), stride=(1, 1), padding=6)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(9, 9), stride=(1, 1), padding=4)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=2)

        self.cbl_s_4 = nn.Sequential(
            CBL(2048, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 512, 1, 1, 0)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        cbl0 = self.cbl_s_0(x)  # (-1, 512, 19, 19)
        p1 = self.max_pool1(cbl0)  # (-1, 512, 19, 19)
        p2 = self.max_pool2(cbl0)  # (-1, 512, 19, 19)
        p3 = self.max_pool3(cbl0)  # (-1, 512, 19, 19)
        tmp = torch.cat((p1, p2, p3, cbl0), dim=1)  # (-1, 2048, 19, 19)
        y19 = self.cbl_s_4(tmp)
        return y19  # (-1, 512, 19, 19)


class Process2(nn.Module):
    def __init__(
            self
    ):
        super().__init__()

        self.cbl_s_0 = nn.Sequential(
            CBL(512, 256, 1, 1, 0),

        )

        self.up_sample_1 = nn.Sequential(
            CBL(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.cbl_s_2 = nn.Sequential(
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
        )

    def forward(
            self,
            x: torch.Tensor,
            y19: torch.Tensor
    ):

        tmp0 = self.cbl_s_0(x)  # (-1, 256, 38, 38)
        tmp1 = self.up_sample_1(y19)  # (-1, 256, 38, 38)
        tmp = torch.cat((tmp0, tmp1), dim=1)    # (-1, 512, 38, 38)
        y38 = self.cbl_s_2(tmp)
        return y38  # (-1, 256, 38, 38)


class Process3(nn.Module):
    def __init__(
            self
    ):
        super().__init__()

        self.cbl_s_0 = nn.Sequential(
            CBL(256, 128, 1, 1, 0),

        )
        self.up_sample_1 = nn.Sequential(
            CBL(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            y38: torch.Tensor
    ):
        tmp0 = self.cbl_s_0(x)  # (-1, 128, 76, 76)
        tmp1 = self.up_sample_1(y38)   # (-1, 128, 76, 76)
        y76 = torch.cat((tmp0, tmp1), dim=1)  # (-1, 256, 76, 76)
        return y76  # (-1, 256, 76, 76)


class Process4(nn.Module):
    def __init__(
            self
    ):
        super().__init__()

        self.y76_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(256, 256, 3, 1, 1),
        )

    def forward(
            self,
            y38: torch.Tensor,
            y76: torch.Tensor,
    ):
        y76_d = self.y76_down(y76)  # (-1, 256, 38, 38)
        y38_a = torch.cat((y76_d, y38), dim=1)   # (-1, 512, 38, 38)
        return y38_a  # (-1, 512, 38, 38)


class Process5(nn.Module):
    def __init__(
            self
    ):
        super().__init__()
        self.y38_a_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            CBL(512, 512, 3, 1, 1),
        )

    def forward(
            self,
            y19: torch.Tensor,
            y38_a: torch.Tensor,
    ):
        y38_a_d = self.y38_a_down(y38_a)  # (-1, 512, 19, 19)
        y19_a = torch.cat((y38_a_d, y19), dim=1)   # (-1, 1024, 19, 19)
        return y19_a  # (-1, 1024, 19, 19)


class Neck(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.process1 = Process1()
        self.process2 = Process2()
        self.process3 = Process3()
        self.process4 = Process4()
        self.process5 = Process5()

    def forward(
            self,
            c3: torch.Tensor,
            c4: torch.Tensor,
            c5: torch.Tensor,
    ):
        y19 = self.process1(c5)
        y38 = self.process2(c4, y19)
        y76 = self.process3(c3, y38)

        y38_a = self.process4(y38, y76)
        y19_a = self.process5(y19, y38_a)
        return y76, y38_a, y19_a


class Head(nn.Module):
    def __init__(
            self,
            num_anchors_for_single_size: int,
            num_classes: int,
    ):
        super().__init__()
        # for y76 (-1, 256, 76, 76)
        self.head1 = nn.Sequential(
            CBL(256, 128, 1, 1, 0),
            CBL(128, 256, 3, 1, 1),
            CBL(256, 128, 1, 1, 0),
            CBL(128, 256, 3, 1, 1),
            CBL(256, 128, 1, 1, 0),
            CBL(128, 256, 3, 1, 1),
            nn.Conv2d(256, num_anchors_for_single_size * (num_classes + 5), 1, 1, 0)
        )

        # for y38_a # (-1, 512, 38, 38)
        self.head2 = nn.Sequential(
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            CBL(512, 256, 1, 1, 0),
            CBL(256, 512, 3, 1, 1),
            nn.Conv2d(512, num_anchors_for_single_size * (num_classes + 5), 1, 1, 0),
        )

        # for y19_a # (-1, 1024, 19, 19)
        self.head3 = nn.Sequential(
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            CBL(1024, 512, 1, 1, 0),
            CBL(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, num_anchors_for_single_size * (num_classes + 5), 1, 1, 0),
        )

    def forward(
            self,
            y76: torch.Tensor,
            y38_a: torch.Tensor,
            y19_a: torch.Tensor
    ):
        y76_o = self.head1(y76)
        y38_o = self.head2(y38_a)
        y19_o = self.head3(y19_a)
        return y76_o, y38_o, y19_o


class YOLOV4Model(DevModel):
    def __init__(
            self,
            net: Union[nn.Module, CSPDarkNet53],
            num_anchors_for_single_size: int = 3,
            num_classes: int = 20,
    ):
        # suggest shape of input: (-1, 3, 608, 608)
        # actually, you could use other size (e.g. 416, make sure size%32==0)
        super().__init__(net)
        """
        net is just backbone
        """

        self.neck = Neck()
        self.head = Head(num_anchors_for_single_size, num_classes)

    def forward(
            self,
            x: torch.Tensor
    ):
        c3, c4, c5 = self.net(x)
        y76, y38_a, y19_a = self.neck(c3, c4, c5)
        y76_o, y38_o, y19_o = self.head(y76, y38_a, y19_a)
        return {
            'for_s': y76_o,  # s=8
            'for_m': y38_o,  # s=16
            'for_l': y19_o,  # s=32
        }


if __name__ == '__main__':
    import time

    img = torch.rand(size=(1, 3, 608, 608))
    backbone_csp_darknet_53 = get_backbone_csp_darknet_53()
    m = YOLOV4Model(
        net=backbone_csp_darknet_53,
        num_classes=20,
        num_anchors_for_single_size=3
    )
    a = time.time()
    y = m(img)
    b = time.time()
    print(b-a)
    for key, val in y.items():
        print('key:{}, val_shape:{}'.format(key, val.shape))
