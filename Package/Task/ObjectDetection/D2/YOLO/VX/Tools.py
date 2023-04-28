from Package.Task.ObjectDetection.D2.Dev import DevTool
from Package.Task.ObjectDetection.D2.YOLO.VX.Typing import *
from abc import abstractmethod
import torch


class YOLOVXTool(DevTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def get_grid_number_and_pre_anchor_w_h(
            image_wh: tuple,
            image_shrink_rate: Union[dict, tuple],
            pre_anchor_w_h_rate: Union[dict, tuple]
    ):
        raise RuntimeError("This method has been discarded!")

    @staticmethod
    @abstractmethod
    def make_target(
            labels: List[LABEL],
            image_size: int = 640,
            image_shrink_rate: Tuple[int, int, int] = (8, 16, 32),
            cls_num: int = 80,
            multi_positives: bool = True,
    ) -> torch.Tensor:
        """

        在早期的 YOLO 系列（V1除外）都有anchor的概念，因此解码以及 制作标签都相当麻烦，涉及 anchor 匹配的问题(因此没有了IOU的计算)
        但是 YOLO vx等后期的 YOLO 系列代码都选择了  anchor-free的做法...

        """
        grid_num_vec = [image_size//rate for rate in image_shrink_rate]
        batch_size = len(labels)
        info_num = 4 + 1 + cls_num
        res_vec = [
            torch.zeros(
                size=(batch_size, info_num, grid_num, grid_num),
                dtype=torch.float32
            ) for grid_num in grid_num_vec
        ]
        offset_cxy = (-1, 0, +1)

        for batch_ind, label in enumerate(labels):
            for _, obj in enumerate(label):
                """
                注意position是box在图像级别的绝对坐标左上，右下
                """
                kind_ind, position = obj
                cx = 1.0 * (position[0] + position[2]) * 0.5
                cy = 1.0 * (position[1] + position[3]) * 0.5
                for rate_ind in range(len(image_shrink_rate)):
                    """
                    需要把图像级别的中心点坐标映射到 不同尺度的特征图中去(也即grid级别的中心点坐标)，
                    """

                    grid_cx = min(max(0, int(cx/image_shrink_rate[rate_ind])), grid_num_vec[rate_ind] - 1)
                    grid_cy = min(max(0, int(cy/image_shrink_rate[rate_ind])), grid_num_vec[rate_ind] - 1)
                    """
                    寻找需要制作标签的索引
                    """
                    need_set_positive_grid_xy = []
                    if multi_positives:
                        """
                        在中心点附近的3*3格子内都设置  positives 标签
                        """
                        for offset_cx in offset_cxy:
                            for offset_cy in offset_cxy:
                                a = min(max(0, grid_cx + offset_cx), grid_num_vec[rate_ind] - 1)
                                b = min(max(0, grid_cy + offset_cy), grid_num_vec[rate_ind] - 1)
                                need_set_positive_grid_xy.append((a, b))
                    else:
                        """
                        只在中心点设置  positives 标签
                        """
                        need_set_positive_grid_xy.append((grid_cx, grid_cy))
                    """
                    制作标签
                    """
                    for x, y in need_set_positive_grid_xy:
                        res_vec[rate_ind][batch_ind, :4, y, x] = torch.tensor(
                            position,
                            dtype=torch.float32
                        )
                        res_vec[rate_ind][batch_ind, 4, y, x] = 1.0
                        res_vec[rate_ind][batch_ind, 5 + kind_ind, y, x] = 1.0

        """
        这里需要和  模型的输出结构保持一致，请注意模型输出的格式
        shape [batch, box_num, info_num]
        其中 box_num 为各个 grid_num * grid_num之和
        """
        res_vec = [
            res.view(batch_size, info_num, -1).permute(0, 2, 1) for res in res_vec
        ]
        return torch.cat(res_vec, dim=1)

    @staticmethod
    @abstractmethod
    def split_target(
            *args,
            **kwargs
    ) -> dict:
        raise RuntimeError("This method has been discarded!")

    @staticmethod
    @abstractmethod
    def split_predict(
            *args,
            **kwargs
    ) -> dict:
        raise RuntimeError("This method has been discarded!")
