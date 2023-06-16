#
# 这里定义的是YOLO VX 模型结构，参考的源代码地址（原文放出的地址）为：https://github.com/Megvii-BaseDetection/YOLOX
# 许多代码选择将训练阶段的损失函数计算，以及推理阶段的box解码都放在model里，我不是特别喜欢

# YOLOVXModel 继承至 DevModel，其中 net 就是back_bone 代表主干网络

import torch
import torch.nn as nn
from typing import *
from Package.Task.ObjectDetection.D2.YOLO.VX.Tools import YOLOVXTool
from Package.Task.ObjectDetection.D2.YOLO.VX.model_based_dark_net import YOLOVXBackBoneBasedDarkNet53, \
    YOLOVXNeckBasedDarkNet53
from Package.Task.ObjectDetection.D2.YOLO.VX.model_original import YOLOVXBackBoneOriginal, YOLOVXNeckOriginal


class YOLOVXModel(nn.Module):
    def __init__(
            self,
            backbone: Union[YOLOVXBackBoneBasedDarkNet53, YOLOVXBackBoneOriginal],
            neck: Union[YOLOVXNeckBasedDarkNet53, YOLOVXNeckOriginal],
            head: nn.ModuleList,
            image_shrink_rate: Sequence[int] = (8, 16, 32),
    ):
        """
        早期参照官方的YOLO X 实现 了一个版本的网络结构，但是后面run的时候 发现这个BackBone很难train（即使是直接用官方code），
        找了很久才发现这个问题
        所以后续找了 YOLO V3的 BackBone （以及配套的Neck） 在 VOC上 比较nice
        """
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.head_list = head

        self.image_shrink_rate = image_shrink_rate
        self.compute_size()

    def compute_size(
            self
    ):
        total = 0
        for i, v in self.named_parameters():
            total += v.numel()
        if total < 1e+6:
            print("\ntotal param : {}\n".format(total))
        elif total < 1e+9:
            print("\ntotal param : {:.2f}M\n".format(1.0 * total / 1e+6))
        else:
            print("\ntotal param : {:.2f}B\n".format(1.0 * total / 1e+9))

    def forward(
            self,
            x: torch.Tensor
    ):
        r"""*
        获得特征  c3 c4 c5
        """
        c3, c4, c5 = self.backbone(x)

        p_3_4_5 = self.neck(c3, c4, c5)
        # for p in p_3_4_5:
        #     print(p.shape)
        o_3_4_5 = []
        for head, c, img_shrink_rate in zip(self.head_list, p_3_4_5, self.image_shrink_rate):
            o = head(c)  # shape [batch, info_num, grid_h, grid_w]
            batch_size, info_num, grid_y_num, grid_x_num = o.shape

            o = o.view(batch_size, info_num, -1).permute(0, 2, 1)  # shape [batch, grid_h*grid_w, info_num]
            grid_ind = YOLOVXTool.get_grid(grid_num=(grid_x_num, grid_y_num))  # shape [grid_h, grid_w, 2]

            grid_ind = grid_ind.view(-1, 2).unsqueeze(dim=0).to(o.device)  # shape [1, grid_h*grid_w, 2]
            r"""
            box decode -->

            这里和YOLO vx 给出的官方解码box代码保证了一致
            o[..., 0:2] 
                解码前是 grid 级别的box左上角 相对于grid index 的偏移量
                解码后是 image 级别的box左上角坐标
            o[..., 2:4]
                解码前是 grid级别的box 的w  和 h
                解码后是 image级别的box 右下角坐标 

            值得注意的是，解码后的box 并不一定是严格意义上 的box，因为有些坐标是负值，这里并没有采用clip操作，主要是不想影响损失函数的
            计算，后续的代码需要注意这一点。
            """
            # o[..., 0:2] = (o[..., 0:2] + grid_ind) * img_shrink_rate  # left_top_x, left_top_y
            # o[..., 2:4] = torch.exp(o[..., 2:4]) * img_shrink_rate + o[..., 0:2]  # right_bottom_x, right_bottom_y

            center_x_y = (2 * torch.sigmoid(o[..., 0:2]) - 1 + grid_ind) * img_shrink_rate  # scaled in image

            """
            注意： 这个wh的解码方式最好是让输出基于grid尺寸
            """
            w_h = torch.sigmoid(o[..., 2:4]) * (grid_x_num + grid_y_num)/2.0 * img_shrink_rate  # scaled in image
            # w_h = torch.exp(o[..., 2:4]) * img_shrink_rate  # scaled in image

            o[..., 0:2] = (center_x_y - 0.5 * w_h)

            o[..., 2:4] = center_x_y + 0.5 * w_h

            o_3_4_5.append(o)

        res = torch.cat(o_3_4_5, dim=1)  # shape [batch, box_num, info_num]
        return res


if __name__ == '__main__':
    from Package.Task.ObjectDetection.D2.YOLO.VX.model_original import *
    from Package.Task.ObjectDetection.D2.YOLO.VX.model_based_dark_net import *
    x_in = torch.rand(size=(1, 3, 640, 640))

    backbone = get_back_bone_based_dark_net_53()
    # neck = get_neck_based_dark_net_53()
    # head = get_head_based_dark_net_53(wide_mul=1, cls_num=20)

    # backbone = get_back_bone_original(
    #     in_channel=3,
    #     base_channel=64,
    #     wide_mul=1,
    #     deep_mul=3
    # )
    neck = get_neck_original(
        base_channel=64,
        wide_mul=1,
        deep_mul=3
    )
    head = get_head_original(
        base_channel=64,
        wide_mul=1,
        cls_num=20
    )
    m = YOLOVXModel(
        backbone,
        neck,
        head
    )
    y_out = m(x_in)
    print(y_out.shape)
