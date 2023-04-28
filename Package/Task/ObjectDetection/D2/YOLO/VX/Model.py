#
# 这里定义的是YOLO VX 模型结构，参考的源代码地址（原文放出的地址）为：https://github.com/Megvii-BaseDetection/YOLOX
# 许多代码选择将训练阶段的损失函数计算，以及推理阶段的box解码都放在model里，我不是特别喜欢

# YOLOVXModel 继承至 DevModel，其中 net 就是back_bone 代表主干网络

import torch
from Package.Task.ObjectDetection.D2.Dev import DevModel
import torch.nn as nn
from typing import *
from Package.Task.ObjectDetection.D2.YOLO.V4.Model import CSPDarkNet53
from Package.Task.ObjectDetection.D2.YOLO.VX.Tools import YOLOVXTool


class YOLOVXDecoupledHead(nn.Module):
    def __init__(
            self,
            in_feature: int,
            feature_mapping: int = 256,
            cls_num: int = 80
    ):
        super().__init__()
        self.__block_1 = nn.Sequential(
            nn.Conv2d(in_feature, feature_mapping, 3, 1, 1),
            nn.BatchNorm2d(feature_mapping),
            nn.Mish(),

            nn.Conv2d(feature_mapping, feature_mapping, 3, 1, 1),
            nn.BatchNorm2d(feature_mapping),
            nn.Mish(),

            nn.Conv2d(feature_mapping, cls_num, 1, 1, 0),
            # nn.Sigmoid(),
            # 这里训练阶段是不需要sigmoid or softmax函数的，因为后续pytorch提供的损失函数集成的有，
            # 当然有时候我们更喜欢叫他 logits，这要求程序员自己要注意，在推理阶段，补上sigmoid等类似的激活操作
        )

        self.__block_2_1 = nn.Sequential(
            nn.Conv2d(in_feature, feature_mapping, 3, 1, 1),
            nn.BatchNorm2d(feature_mapping),
            nn.Mish(),

            nn.Conv2d(feature_mapping, feature_mapping, 3, 1, 1),
            nn.BatchNorm2d(feature_mapping),
            nn.Mish(),

        )
        self.__block_2_2_for_obj = nn.Sequential(
            nn.Conv2d(feature_mapping, 1, 1, 1, 0),
            # nn.Sigmoid(),
            # 注释掉的理由同上
        )
        self.__block_2_2_for_pos = nn.Sequential(
            nn.Conv2d(feature_mapping, 4, 1, 1, 0),
        )
        self.__cls_num = cls_num

    def forward(
            self,
            x: torch.Tensor
    ):
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


class YOLOVXPanFpn(nn.Module):
    def __init__(
            self,
    ):
        """
        YOLO vx的主干网络，主要负责特征提取和路径聚合，这里直接使用CSP darknet 53
        """
        super().__init__()
        self.__net = CSPDarkNet53()

    def forward(
            self,
            x: torch.Tensor
    ):
        return self.__net(x)


class YOLOVXModel(DevModel):
    def __init__(
            self,
            back_bone: Union[nn.Module, YOLOVXPanFpn],
            feature_channel: Sequence[int] = (256, 512, 1024),
            feature_mapping: int = 256,
            cls_num: int = 80,
            image_shrink_rate: Sequence[int] = (8, 16, 32),
    ):
        r"""
        这个版本的YOLO X 实现的是 csp_darknet_53 版本， 因此backbone 返回的特征C3、C4、C5的通道数分别为：256， 512， 1024
        如果使用的是其他的backbone，请确保 feature_channel 和 自定义的 backbone 保持一致

        值得注意的是，很多研究人员的代码选择将不同特征的输出/预测整合在一起，在我之前的YOLO系列代码中并没有选择这种做法，
        主要是因为那样的做法会导致代码的可读性变得很差。因此在我之前代码里，总会看到返回值是一个包含原始输出的字典

        当然！如果考虑到项目后续要实际部署，当然还是建议 把解码之后的结果返回

        由于 YOLO 的一些较新的作品都采用了iou loss，因此，后续我的代码里将直接解码box
        """
        super().__init__(back_bone)
        self.head_list = nn.ModuleList(
            YOLOVXDecoupledHead(channel, feature_mapping, cls_num) for channel in feature_channel
        )
        self.image_shrink_rate = image_shrink_rate

    def forward(
            self,
            x: torch.Tensor
    ):
        r"""*
        获得 csp darknet 53的特征  c3 c4 c5
        """
        c_3_4_5 = self.net(x)
        o_3_4_5 = []
        for head, c, img_shrink_rate in zip(self.head_list, c_3_4_5, self.image_shrink_rate):
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
            o[..., 2:0]
                解码前是 grid级别的box 的w  和 h
                解码后是 image级别的box 右下角坐标 
            
            值得注意的是，解码后的box 并不一定是严格意义上 的box，因为有些坐标是负值，这里并没有采用clip操作，主要是不想影响损失函数的
            计算，后续的代码需要注意这一点。
            """
            o[..., 0:2] = (o[..., 0:2] + grid_ind) * img_shrink_rate  # left_top_x, left_top_y
            o[..., 2:4] = torch.exp(o[..., 2:4]) * img_shrink_rate + o[..., 0:2]  # right_bottom_x, right_bottom_y

            o_3_4_5.append(o)

        res = torch.cat(o_3_4_5, dim=1)  # shape [batch, box_num, info_num]
        return res

