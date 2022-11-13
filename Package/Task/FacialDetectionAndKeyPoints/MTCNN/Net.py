import torch
import torch.nn as nn
from abc import abstractmethod
from collections import OrderedDict


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class BaseNet(nn.Module):
    def __init__(
            self,
            ):
        super().__init__()
        self._init_net()
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(
            m
    ):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)
            nn.init.constant(m.bias, 0.1)

    @abstractmethod
    def _init_net(self):
        raise NotImplementedError(
            'Please override this method (_init_net).'
        )

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            'Please override this method (forward).'
        )


class PNet(BaseNet):
    def __init__(
            self,
    ):
        super().__init__()

    def _init_net(
            self,
    ):
        # backend
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, kernel_size=3, stride=1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),
            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),
            ('conv3', nn.Conv2d(16, 32, kernel_size=3, stride=1)),
            ('prelu3', nn.PReLU(32))
        ]))

        # detection
        self.cls = nn.Sequential(OrderedDict([
            ('conv4-1', nn.Conv2d(32, 2, kernel_size=1, stride=1)),
            # ('softmax', nn.Softmax(1))
        ]))
        # bounding box regresion
        self.box_offset = nn.Sequential(OrderedDict([
            ('conv4-2', nn.Conv2d(32, 4, kernel_size=1, stride=1)),
        ]))

        # landmark regression
        self.landmarks = nn.Sequential(OrderedDict([
            ('conv4-2', nn.Conv2d(32, 10, kernel_size=1, stride=1))
        ]))

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ):
        feature_map = self.body(x)
        label = self.cls(feature_map)
        offset = self.box_offset(feature_map)
        landmarks = self.landmarks(feature_map)

        return {
            'cls': label,
            'pos_offset': offset,
            'key_point': landmarks
        }


class RNet(BaseNet):
    def __init__(self):
        super().__init__()

    def _init_net(self):

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, kernel_size=3, stride=1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, kernel_size=3, stride=1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, kernel_size=2, stride=1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        # detection
        self.cls = nn.Sequential(OrderedDict([
            ('conv5-1', nn.Linear(128, 2)),
            # ('softmax', nn.Softmax(1))
        ]))
        # bounding box regression
        self.box_offset = nn.Sequential(OrderedDict([
            ('conv5-2', nn.Linear(128, 4))
        ]))

        # lanbmark localization
        self.landmarks = nn.Sequential(OrderedDict([
            ('conv5-3', nn.Linear(128, 10))
        ]))

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ):
        # backend
        x = self.body(x)

        # detection
        det = self.cls(x)
        box = self.box_offset(x)
        landmarks = self.landmarks(x)

        return {
            'cls': det,
            'pos_offset': box,
            'key_point': landmarks
        }


class ONet(BaseNet):
    def __init__(self):
        super().__init__()

    def _init_net(self):
        # backend

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, kernel_size=2, stride=1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        # detection
        self.cls = nn.Sequential(OrderedDict([
            ('conv6-1', nn.Linear(256, 2)),
            # ('softmax', nn.Softmax(1))
        ]))
        # bounding box regression
        self.box_offset = nn.Sequential(OrderedDict([
            ('conv6-2', nn.Linear(256, 4))
        ]))
        # lanbmark localization
        self.landmarks = nn.Sequential(OrderedDict([
            ('conv6-3', nn.Linear(256, 10))
        ]))

    def forward(
            self,
            x: torch.Tensor,
            *args,
            **kwargs
    ):
        # backend
        x = self.body(x)

        # detection
        det = self.cls(x)

        # box regression
        box = self.box_offset(x)

        # landmarks regresion
        landmarks = self.landmarks(x)

        return {
            'cls': det,
            'pos_offset': box,
            'key_point': landmarks
        }


if __name__ == "__main__":
    p_net = PNet()
    r_net = RNet()
    o_net = ONet()

    x = torch.rand(size=(128, 3, 156, 48))
    o: dict = p_net(x)
    print('p-net:')
    for key, val in o.items():
        print("\t{} --> shape: {}".format(key, val.shape))

    x = torch.rand(size=(128, 3, 24, 24))
    o: dict = r_net(x)
    print('r-net:')
    for key, val in o.items():
        print("\t{} --> shape: {}".format(key, val.shape))

    x = torch.rand(size=(128, 3, 48, 48))
    o: dict = o_net(x)
    print('o-net:')
    for key, val in o.items():
        print("\t{} --> shape: {}".format(key, val.shape))
