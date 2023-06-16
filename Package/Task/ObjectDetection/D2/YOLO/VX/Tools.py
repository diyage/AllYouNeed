from Package.Task.ObjectDetection.D2.Dev import DevTool
from Package.Task.ObjectDetection.D2.YOLO.VX.Typing import *
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IOUMoreToMore(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_iou_one_to_more(
            gt: torch.Tensor,
            pre_es: torch.Tensor,
    ) -> torch.Tensor:
        # gt shape --> [4, ]
        # pre_es shape --> [pre_box_num, 4]
        gt_expand = gt.unsqueeze(0).expand_as(pre_es)
        iou = YOLOVXTool.iou(
            gt_expand,
            pre_es
        )
        del gt_expand
        return iou.view(-1, )

    def forward(
            self,
            gt_boxes: torch.Tensor,
            pre_boxes: torch.Tensor,
    ):
        iou_vec = []
        for i in range(gt_boxes.shape[0]):
            iou_vec.append(
                self.compute_iou_one_to_more(
                    gt_boxes[i],
                    pre_boxes
                )
            )

        return torch.stack(iou_vec, dim=0)  # [gt_box_num, pre_box_num]


class CrossEntropyMoreToMore(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ce = nn.CrossEntropyLoss(reduction='none')
        self.ce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
            self,
            gt_kind_ind: torch.Tensor,
            pre_cls: torch.Tensor
    ):
        res_vec = []
        for i in range(gt_kind_ind.shape[0]):
            g = gt_kind_ind[i].unsqueeze(0).expand(size=(pre_cls.shape[0], ))
            target = F.one_hot(g, num_classes=pre_cls.shape[1]).float().to(pre_cls.device)
            loss = self.ce(
                pre_cls,
                target
            )
            res_vec.append(loss.sum(dim=-1))
            del target

        return torch.stack(res_vec, dim=0)  # [gt_box_num, pre_box_num]


class IsInBoxAndCenterBox(nn.Module):
    def __init__(
            self,
            radius: float = 2.5,
            img_size: int = 640,
            img_shrink_rate: Tuple[int, int, int] = (8, 16, 32)
    ):
        super().__init__()
        anchor_point_on_image = []
        radius_on_image = []

        for rate in img_shrink_rate:
            num = img_size//rate
            grid_ind = YOLOVXTool.get_grid(grid_num=(num, num))  # shape [grid_h, grid_w, 2]
            grid_ind = grid_ind.view(-1, 2)  # shape [grid_h*grid_w, 2]
            anchor_point_on_image.append(grid_ind*rate)
            radius_on_image.extend([radius * rate] * grid_ind.shape[0])

        self.img_size: int = 640

        self.anchor_point_on_image = torch.cat(
            anchor_point_on_image,
            dim=0
        ).float()
        # (pre_box_num, 2)
        self.radius_on_image = torch.tensor(radius_on_image, dtype=torch.float32)
        # (pre_box_num, )
        self.radius_on_image = self.radius_on_image.unsqueeze(-1).expand_as(
            self.anchor_point_on_image
        )
        # (pre_box_num, 2)

    @staticmethod
    def point_in_box(
            points: torch.Tensor,  # (box_num, 2)
            boxes: torch.Tensor  # (box_num, 4)
    ) -> torch.Tensor:

        delta_left_top = points - boxes[:, :2]
        delta_right_bottom = boxes[:, 2:] - points
        delta = torch.cat([delta_left_top, delta_right_bottom], dim=-1)

        is_in_box = delta.min(dim=-1)[0] > 0
        return is_in_box.float()

    def forward(
            self,
            gt_boxes: torch.Tensor,  # [gt_box_num, 4]
            pre_boxes: torch.Tensor  # [pre_box_num, 4]
    ):

        res_vec = []
        self.anchor_point_on_image = self.anchor_point_on_image.to(gt_boxes.device)
        self.radius_on_image = self.radius_on_image.to(gt_boxes.device)

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:])*0.5
        # [gt_box_num, 2]

        """
        注意是判断pre_boxes所在的anchor point 与 gt_boxes的匹配关系，而不是pre_boxes本身,
        事实上 这个pre_boxes传递的没有意义，但为了统一，还是留下了这个参数
        """
        for i in range(gt_boxes.shape[0]):
            # 判断 各个 anchor point 是否落在
            # 该 gt box里
            anchor_point_in_gt_box = self.point_in_box(
                self.anchor_point_on_image,
                gt_boxes[i].unsqueeze(0).expand_as(pre_boxes)
            )  # (pre_box_num, )

            # 判断 各个 anchor point 是否落在
            # 以 gt box 的center为中心， radius_on_image为半径的 正方形框里
            center = gt_centers[i].unsqueeze(0).expand(size=(
                pre_boxes.shape[0],
                2
            ))
            # (pre_box_num, 2)

            x1y1 = center - self.radius_on_image
            x2y2 = center + self.radius_on_image

            gt_center_box = torch.cat([x1y1, x2y2], dim=-1).clamp(0, self.img_size)
            # (pre_box_num, 4)

            anchor_point_in_gt_center_box = self.point_in_box(
                self.anchor_point_on_image,
                gt_center_box
            )
            # (pre_box_num, )
            is_in_box_and_center = anchor_point_in_gt_box * anchor_point_in_gt_center_box
            res_vec.append(is_in_box_and_center)

        return torch.stack(
            res_vec,
            dim=0
        )


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
    @torch.no_grad()
    def sim_ota_make_target(
            outputs: torch.Tensor,
            labels: List[LABEL],
            image_size: int = 640,
            image_shrink_rate: Tuple[int, int, int] = (8, 16, 32),
            radius: float = 2.5,
            dynamic_k: int = 10,
            weight_of_position: float = 3.0,
            weight_of_center: float = 1e+6
    ):

        def compute_pair_wise_and_is_same_center(
                now_img_output: torch.Tensor,
                now_img_label: LABEL
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            #######################################################################
            pre_boxes = now_img_output[:, :4]

            gt_boxes = [obj[1] for obj in now_img_label]
            gt_boxes = torch.tensor(gt_boxes, dtype=pre_boxes.dtype).to(pre_boxes.device)

            pair_iou = pair_iou_computer(gt_boxes, pre_boxes)
            #######################################################################

            #######################################################################
            pre_cls = now_img_output[:, 5:]
            gt_kind_ind = [obj[0] for obj in now_img_label]
            gt_kind_ind = torch.tensor(gt_kind_ind, dtype=torch.long).to(pre_cls.device)

            pair_cls = pair_cls_computer(gt_kind_ind, pre_cls)
            #######################################################################

            pair_is_same_center = pair_is_same_center_computer(gt_boxes, pre_boxes)

            #######################################################################
            del gt_boxes, gt_kind_ind
            return pair_iou, pair_cls, pair_is_same_center

        # outputs = outputs.detach().cpu()
        assert len(outputs.shape) == 3
        batch_size, pre_box_num, info_num = outputs.shape

        pair_iou_computer = IOUMoreToMore()
        pair_cls_computer = CrossEntropyMoreToMore()
        # pair_is_same_center_computer = IsSameCenterMoreToMore(radius_on_image)
        pair_is_same_center_computer = IsInBoxAndCenterBox(
            radius,
            img_size=image_size,
            img_shrink_rate=image_shrink_rate
        )

        res = torch.zeros(size=outputs.shape, dtype=torch.float32)

        for batch_ind in range(batch_size):

            gt_box_num = len(labels[batch_ind])

            if gt_box_num == 0:
                continue

            pair_wise_iou, pair_wise_cls_loss, is_same_center = compute_pair_wise_and_is_same_center(
                outputs[batch_ind],
                labels[batch_ind]
            )
            # shape : [gt_box_num, pre_box_num]

            cost = pair_wise_cls_loss + \
                weight_of_position * (1.0 - pair_wise_iou) + \
                weight_of_center * (1.0 - is_same_center)

            dk = min(dynamic_k, pre_box_num)

            top_k_iou, _ = torch.topk(
                pair_wise_iou,
                dk,
                dim=1
            )
            #  (gt_box_num, dk)

            k_for_each_gt_box = torch.clamp(
                top_k_iou.sum(dim=1).int(),
                min=1
            )
            for gt_box_ind in range(gt_box_num):
                kind_ind, position = labels[batch_ind][gt_box_ind]
                k = k_for_each_gt_box[gt_box_ind].item()
                zz, matched_pre_box_ind_vec = torch.topk(
                    cost[gt_box_ind],
                    k,
                    dim=-1,
                    largest=False
                )

                res[batch_ind, matched_pre_box_ind_vec, :4] = torch.tensor(
                    position,
                    dtype=torch.float32
                ).unsqueeze(0).expand(size=(len(matched_pre_box_ind_vec), 4)).to(res.device)
                res[batch_ind, matched_pre_box_ind_vec, 4] = 1.0
                res[batch_ind, matched_pre_box_ind_vec, 5 + kind_ind] = 1.0

                del matched_pre_box_ind_vec

            del pair_wise_iou, pair_wise_cls_loss, is_same_center, cost, top_k_iou, k_for_each_gt_box

        del pair_iou_computer, pair_cls_computer, pair_is_same_center_computer

        torch.cuda.empty_cache()
        return res

    @staticmethod
    def match_which_gird(
            position: Tuple[float, float, float, float],
            width_height_center
    ) -> int:
        # 注意！这个部分 其实完全可以放在data set 里做, 直接用 k mean做
        w = position[2] - position[0]
        h = position[3] - position[1]
        dis = []
        for w_, h_ in width_height_center:
            dis.append(
                (w-w_)**2 + (h-h_)**2
            )
        return np.argmin(dis)

    # @staticmethod
    # @abstractmethod
    # def make_target(
    #         labels: List[LABEL],
    #         width_height_center,
    #         image_size: int = 640,
    #         image_shrink_rate: Tuple[int, int, int] = (8, 16, 32),
    #         cls_num: int = 80,
    #         multi_positives: bool = True,
    # ) -> torch.Tensor:
    #     """
    #
    #     在早期的 YOLO 系列（V1除外）都有anchor的概念，因此解码以及 制作标签都相当麻烦，涉及 anchor 匹配的问题
    #     但是 YOLO vx等后期的 YOLO 系列代码都选择了  anchor-free的做法...
    #     这个make_target 不是 OTA/SimOTA策略，只是为了快速 run通YOLO VX， 而采用的一个静态标签分配策略
    #     将用于后续OTA/SimOTA策略的涨点对比
    #     """
    #     grid_num_vec = [image_size//rate for rate in image_shrink_rate]
    #     batch_size = len(labels)
    #     info_num = 4 + 1 + cls_num
    #     res_vec = [
    #         torch.zeros(
    #             size=(batch_size, info_num, grid_num, grid_num),
    #             dtype=torch.float32
    #         ) for grid_num in grid_num_vec
    #     ]
    #     """
    #     希望box所在中心点的 3*3 grid都认为和这个box匹配成功
    #     """
    #     offset_cxy = (-1, 0, +1)
    #     # offset_cxy = (0, )
    #     for batch_ind, label in enumerate(labels):
    #         for _, obj in enumerate(label):
    #             """
    #             注意position是box在图像级别的绝对坐标左上，右下
    #             """
    #             kind_ind, position = obj
    #             cx = 1.0 * (position[0] + position[2]) * 0.5
    #             cy = 1.0 * (position[1] + position[3]) * 0.5
    #             """
    #             这里考虑了box到底和哪个scale尺度的grid对应，其实就是考虑box到底是小目标、中目标还是大目标
    #             """
    #             rate_ind = YOLOVXTool.match_which_gird(
    #                 position,
    #                 width_height_center=width_height_center
    #             )
    #             if rate_ind < len(image_shrink_rate):
    #
    #             # for rate_ind in range(len(image_shrink_rate)):
    #                 """
    #                 需要把图像级别的中心点坐标映射到 不同尺度的特征图中去(也即grid级别的中心点坐标)，
    #                 """
    #
    #                 grid_cx = min(max(0, int(cx/image_shrink_rate[rate_ind])), grid_num_vec[rate_ind] - 1)
    #                 grid_cy = min(max(0, int(cy/image_shrink_rate[rate_ind])), grid_num_vec[rate_ind] - 1)
    #                 """
    #                 寻找需要制作标签的索引
    #                 """
    #                 need_set_positive_grid_xy = []
    #                 if multi_positives:
    #                     """
    #                     在中心点附近的3*3格子内都设置  positives 标签
    #                     """
    #                     for offset_cx in offset_cxy:
    #                         for offset_cy in offset_cxy:
    #                             a = min(max(0, grid_cx + offset_cx), grid_num_vec[rate_ind] - 1)
    #                             b = min(max(0, grid_cy + offset_cy), grid_num_vec[rate_ind] - 1)
    #                             need_set_positive_grid_xy.append((a, b))
    #                 else:
    #                     """
    #                     只在中心点设置  positives 标签
    #                     """
    #                     need_set_positive_grid_xy.append((grid_cx, grid_cy))
    #                 """
    #                 制作标签
    #                 """
    #                 for x, y in need_set_positive_grid_xy:
    #                     res_vec[rate_ind][batch_ind, :4, y, x] = torch.tensor(
    #                         position,
    #                         dtype=torch.float32
    #                     )
    #                     res_vec[rate_ind][batch_ind, 4, y, x] = 1.0
    #                     res_vec[rate_ind][batch_ind, 5 + kind_ind, y, x] = 1.0
    #
    #     """
    #     这里需要和  模型的输出结构保持一致，请注意模型输出的格式
    #     shape [batch, box_num, info_num]
    #     其中 box_num 为各个 grid_num * grid_num之和
    #     """
    #     res_vec = [
    #         res.view(batch_size, info_num, -1).permute(0, 2, 1) for res in res_vec
    #     ]
    #     return torch.cat(res_vec, dim=1)

    @staticmethod
    @abstractmethod
    def make_target(
            labels: List[LABEL],
            width_height_center,
            image_size: int = 640,
            image_shrink_rate: Tuple[int, int, int] = (8, 16, 32),
            cls_num: int = 80,
            multi_positives: bool = True,
    ) -> torch.Tensor:
        """

        在早期的 YOLO 系列（V1除外）都有anchor的概念，因此解码以及 制作标签都相当麻烦，涉及 anchor 匹配的问题
        但是 YOLO vx等后期的 YOLO 系列代码都选择了  anchor-free的做法...
        这个make_target 不是 OTA/SimOTA策略，只是为了快速 run通YOLO VX， 而采用的一个静态标签分配策略
        将用于后续OTA/SimOTA策略的涨点对比
        """
        grid_num_vec = [image_size // rate for rate in image_shrink_rate]
        batch_size = len(labels)
        info_num = 4 + 1 + cls_num
        res_vec = [
            torch.zeros(
                size=(batch_size, info_num, grid_num, grid_num),
                dtype=torch.float32
            ) for grid_num in grid_num_vec
        ]

        for batch_ind, label in enumerate(labels):
            for _, obj in enumerate(label):
                """
                注意position是box在图像级别的绝对坐标左上，右下
                """
                kind_ind, position = obj
                rate_ind = YOLOVXTool.match_which_gird(
                    position,
                    width_height_center=width_height_center
                )
                if rate_ind < len(image_shrink_rate):
                # for rate_ind in range(len(image_shrink_rate)):
                    position_g = []
                    for p in position:
                        p_g = int(p // image_shrink_rate[rate_ind])
                        p_g = min(max(0, p_g), grid_num_vec[rate_ind] - 1)
                        position_g.append(p_g)
                    x1_g, y1_g, x2_g, y2_g = position_g
                    """
                    寻找需要制作标签的索引
                    """

                    res_vec[rate_ind][batch_ind, :4, y1_g:y2_g+1, x1_g:x2_g+1] = torch.tensor(
                        position,
                        dtype=torch.float32
                    ).unsqueeze(-1).unsqueeze(-1).expand(size=(4, y2_g-y1_g+1, x2_g-x1_g+1))

                    res_vec[rate_ind][batch_ind, 4, y1_g:y2_g+1, x1_g:x2_g+1] = 1.0
                    res_vec[rate_ind][batch_ind, 5 + kind_ind, y1_g:y2_g+1, x1_g:x2_g+1] = 1.0

                    # for x in range(x1_g, x2_g+1):
                    #     for y in range(y1_g, y2_g+1):
                    #         set_this_grid = False
                    #         if np.random.random() > 0.5:
                    #             if res_vec[rate_ind][batch_ind, 4, y, x].item() != 1.0:
                    #                 set_this_grid = True
                    #             else:
                    #                 cgx = x * image_shrink_rate[rate_ind]
                    #                 cgy = y * image_shrink_rate[rate_ind]
                    #                 ##############################################
                    #                 x1, y1, x2, y2 = position
                    #                 cx = 0.5 * (x1 + x2)
                    #                 cy = 0.5 * (y1 + y2)
                    #                 dis_now = (cx - cgx) ** 2 + \
                    #                           (cy - cgy) ** 2
                    #                 ##############################################
                    #                 ##############################################
                    #                 x1, y1, x2, y2 = res_vec[rate_ind][batch_ind, :4, y, x].numpy().tolist()
                    #                 cx = 0.5 * (x1 + x2)
                    #                 cy = 0.5 * (y1 + y2)
                    #                 dis_already = (cx - cgx) ** 2 + \
                    #                               (cy - cgy) ** 2
                    #                 ##############################################
                    #                 if dis_now < dis_already:
                    #                     set_this_grid = True
                    #
                    #         if set_this_grid:
                    #             res_vec[rate_ind][batch_ind, :4, y, x] = torch.tensor(
                    #                 position,
                    #                 dtype=torch.float32
                    #             )
                    #             res_vec[rate_ind][batch_ind, 4, y, x] = 1.0
                    #             res_vec[rate_ind][batch_ind, 5 + kind_ind, y, x] = 1.0

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
    def get_wh_center(
            train_loader,
            n_clusters: int = 3
    ):
        from tqdm import tqdm
        wh = []
        for _, (x, l) in enumerate(tqdm(train_loader, position=0)):
            for labels in l:
                for obj in labels:
                    position = obj[1]
                    wh.append([
                        position[2] - position[0],
                        position[3] - position[1]
                    ])
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters)
        km.fit(wh)
        return km.cluster_centers_

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
