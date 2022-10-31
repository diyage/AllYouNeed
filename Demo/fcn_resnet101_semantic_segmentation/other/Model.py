from Package.Task.Segmentation.SemanticSegmentation.D2.Dev import DevModel
import torch
import torchvision.models as models
import torch.nn as nn
from typing import Union


class FCNResNet101Net(nn.Module):
    def __init__(
            self,
            pre_trained: bool = False,
            num_classes: int = 21,
    ):
        super().__init__()
        self.__model = models.segmentation.fcn_resnet101(
            pretrained=pre_trained,
            num_classes=num_classes
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        res = self.__model(x)
        return res['out']


def get_fcn_resnet101(
        pre_trained: bool = False,
        num_classes: int = 21,
) -> FCNResNet101Net:
    return FCNResNet101Net(
        pre_trained,
        num_classes,
    )


class FCNResNet101Model(DevModel):
    def __init__(
            self,
            net: Union[nn.Module, FCNResNet101Net]
    ):
        super().__init__(net)

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ):
        return self.net(x)


def de_bug_model():
    x = torch.rand(size=(1, 3, 448, 448))
    m = get_fcn_resnet101(True, 21)
    y = m(x)
    # print(m)
    print(y.shape)
    print(y)


if __name__ == '__main__':
    de_bug_model()
