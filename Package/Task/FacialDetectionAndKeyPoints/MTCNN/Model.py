from ..Dev import DevModel
import torch
import torch.nn as nn


class MTCNNModel(DevModel):
    def __init__(
            self,
            p_net: nn.Module,
            r_net: nn.Module,
            o_net: nn.Module
    ):
        super().__init__()
        self.net_map: dict = {
            'p': p_net,
            'r': r_net,
            'o': o_net
        }

    def train(self, mode: bool = True):
        self.training = mode
        for _, net in self.net_map.items():
            net.train(mode)

    def eval(self):
        self.train(mode=False)

    def forward(
            self,
            x: torch.Tensor,
            use_net_type: str,
            *args,
            **kwargs
    ):
        if self.training:
            assert use_net_type in ['p', 'r', 'o']
            out: dict = self.net_map[use_net_type](x)
            return {
                '{}_out'.format(use_net_type): out
            }
        else:
            raise RuntimeError(
                'Please not use __call__ when inference for MTCNNModel(it is too complex).' +
                'Using MTCNNPredictor may be a good idea.'
            )

