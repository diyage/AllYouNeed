"""
This packet is the most-most-most core development tool.
It will serve for all other development tools.
You could use it to define everything !!!
"""
import torch
import numpy as np
from typing import Union
from abc import abstractmethod
from Package.BaseDev import BaseTool


class DevTool(BaseTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def get_grid_number_and_pre_anchor_w_h(
            image_wh: tuple,
            image_shrink_rate: Union[dict, tuple],
            pre_anchor_w_h_rate: Union[dict, tuple]
    ):
        pass

    @staticmethod
    @abstractmethod
    def make_target(
            *args,
            **kwargs
    ):
        """
        create target used for computing loss.
        You may have one question: where is make_predict ?
        Method --make_predict-- is just __call__(or forward, little incorrect) of nn.Module !!!
        So predict is just the output of model(nn.Module you define).
        Please see model.BaseModel
        """
        pass

    @staticmethod
    @abstractmethod
    def split_target(
            *args,
            **kwargs
    ) -> dict:
        """
        split target(created by method make_target).
        you will get result like this:
            {
                 'position': xxx,
                 'conf': xxx,
                 'cls_prob': xxx,
                 ...
            }
        or:
            {
                'key_0': {
                             'position': xxx,
                             'conf': xxx,
                             'cls_prob': xxx,
                             ...
                        },
                'key_1': {
                             'position': xxx,
                             'conf': xxx,
                             'cls_prob': xxx,
                             ...
                        },
                ...
            }
        """
        pass

    @staticmethod
    @abstractmethod
    def split_predict(
            *args,
            **kwargs
    ) -> dict:
        """
        split predict(create by model, i.e. model output).
                you will get result like this:
                    {
                         'position': xxx,
                         'conf': xxx,
                         'cls_prob': xxx,
                         ...
                    }
                or:
                    {
                        'key_0': {
                                     'position': xxx,
                                     'conf': xxx,
                                     'cls_prob': xxx,
                                     ...
                                },
                        'key_1': {
                                     'position': xxx,
                                     'conf': xxx,
                                     'cls_prob': xxx,
                                     ...
                                },
                        ...
                    }
        """
        pass

    @staticmethod
    def get_grid(
            grid_number: tuple
    ):
        index = torch.tensor(list(range(grid_number[0])), dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(index, index)
        grid = torch.cat((grid_c.unsqueeze(-1), grid_r.unsqueeze(-1)), dim=-1)
        # H * W * 2
        return grid

    @staticmethod
    def iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)

        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box
        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)
        return torch.clamp(iou, 0, 1)

    @staticmethod
    def g_iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)
        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box

        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)

        outer_x0y0 = torch.min(box_a_x0y0, box_b_x0y0)
        outer_x1y1 = torch.max(box_a_x1y1, box_b_x1y1)
        outer_is_box = (outer_x1y1 > outer_x0y0).type(outer_x1y1.type()).prod(dim=-1)
        s_outer = torch.prod(outer_x1y1 - outer_x0y0, dim=-1) * outer_is_box
        s_rate = (s_outer - union) / (s_outer + 1e-20)

        g_iou = iou - s_rate
        return torch.clamp(g_iou, -1, 1)

    @staticmethod
    def d_iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)
        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box

        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)

        outer_x0y0 = torch.min(box_a_x0y0, box_b_x0y0)
        outer_x1y1 = torch.max(box_a_x1y1, box_b_x1y1)

        box_a_center = (box_a_x0y0 + box_a_x1y1) * 0.5
        box_b_center = (box_b_x0y0 + box_b_x1y1) * 0.5
        distance_center = torch.sum((box_b_center - box_a_center) ** 2, dim=-1)
        distance_outer = torch.sum((outer_x1y1 - outer_x0y0) ** 2, dim=-1)
        distance_rate = distance_center / (distance_outer + 1e-20)

        d_iou = iou - distance_rate
        return torch.clamp(d_iou, -1, 1)

    @staticmethod
    def c_iou(
            box_a: torch.Tensor,
            box_b: torch.Tensor
    ):
        # box_a/box_b (..., 4)
        box_a_x0y0 = box_a[..., :2]
        box_a_x1y1 = box_a[..., 2:]

        box_b_x0y0 = box_b[..., :2]
        box_b_x1y1 = box_b[..., 2:]

        s_a = torch.prod(box_a_x1y1 - box_a_x0y0, dim=-1)
        s_b = torch.prod(box_b_x1y1 - box_b_x0y0, dim=-1)

        inter_x0y0 = torch.max(box_a_x0y0, box_b_x0y0)
        inter_x1y1 = torch.min(box_a_x1y1, box_b_x1y1)
        inter_is_box = (inter_x1y1 > inter_x0y0).type(inter_x1y1.type()).prod(dim=-1)
        s_inter = torch.prod(inter_x1y1 - inter_x0y0, dim=-1) * inter_is_box

        union = s_a + s_b - s_inter
        iou = s_inter / (union + 1e-20)

        outer_x0y0 = torch.min(box_a_x0y0, box_b_x0y0)
        outer_x1y1 = torch.max(box_a_x1y1, box_b_x1y1)

        box_a_center = (box_a_x0y0 + box_a_x1y1) * 0.5
        box_b_center = (box_b_x0y0 + box_b_x1y1) * 0.5
        distance_center = torch.sum((box_b_center - box_a_center) ** 2, dim=-1)
        distance_outer = torch.sum((outer_x1y1 - outer_x0y0) ** 2, dim=-1)
        distance_rate = distance_center / (distance_outer + 1e-20)

        box_a_wh = box_a_x1y1 - box_a_x0y0
        box_b_wh = box_b_x1y1 - box_b_x0y0
        w2 = box_b_wh[..., 0]
        h2 = box_b_wh[..., 1]
        w1 = box_a_wh[..., 0]
        h1 = box_a_wh[..., 1]

        v = (4 / (np.pi ** 2)) * torch.pow((torch.atan(w2 / (h2 + 1e-20)) - torch.atan(w1 / (h1 + 1e-20))), 2)
        alpha = v / (1 - iou + v + 1e-20)  # type: torch.Tensor

        # c_iou = iou - (distance_rate + alpha.detach() * v)
        c_iou = iou - (distance_rate + alpha * v)
        return torch.clamp(c_iou, -1, 1)
