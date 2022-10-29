from Package.Task.FacialRecognition.D2.Dev import DevLoss
import torch
import torch.nn as nn
"""
Be careful here!
In ArcFace, its idea could be implemented as ArcFaceHead(model) or ArcFaceLoss(loss).
While, we chose the first one. 
So, ArcFaceLoss is just pure CrossEntropyLoss...
"""


class ArcFaceLoss(DevLoss):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
            self,
            predict: dict,
            target: torch.Tensor,
            *args,
            **kwargs
    ):
        loss_dict = {
            'total_loss': 0.0,

        }
        out = predict.get('out')  # logits
        loss_dict['total_loss'] += self.ce(out, target)
        return loss_dict
