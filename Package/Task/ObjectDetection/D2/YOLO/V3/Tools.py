import torch
import numpy as np
from Package.Task.ObjectDetection.D2.Dev import DevTool
from typing import Union


class YOLOV3Tool(DevTool):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_grid_number_and_pre_anchor_w_h(
            image_wh: tuple,
            image_shrink_rate: dict,
            pre_anchor_w_h_rate: dict
    ):

        grid_number = {}
        pre_anchor_w_h = {}

        for anchor_key, shrink_rate in image_shrink_rate.items():
            anchor_rate = pre_anchor_w_h_rate.get(anchor_key)

            single_grid_number = (
                image_wh[0] // shrink_rate[0],
                image_wh[1] // shrink_rate[1]
            )
            single_pre_anchor = tuple([
                (rate[0] * single_grid_number[0], rate[1] * single_grid_number[1]) for rate in anchor_rate
            ])
            grid_number[anchor_key] = single_grid_number
            pre_anchor_w_h[anchor_key] = single_pre_anchor

        return grid_number, pre_anchor_w_h

    @staticmethod
    def split_predict(
            out_put: dict,
            anchor_number_for_single_size: int,
    ):
        res = {}
        """
        res --->
        
        # key : 'for_m', 'for_s', 'for_l'
        # val : dict --> {'position': xxx, 'conf': xxx, 'cls_prob': xxx}
        
        """
        for key, x in out_put.items():

            N, C, H, W = x.shape
            K = C // anchor_number_for_single_size  # K = (x, y, w, h, conf, kinds0, kinds1, ...)
            # C = anchor_number * K
            x = x.view(N, anchor_number_for_single_size, K, H, W)
            x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

            position = [x[..., 0:4], None]
            conf = x[..., 4]  # N * H * W * a_n
            cls_prob = x[..., 5:]  # N * H * W * a_n * ...

            now_size_res = {
                'position': position,  # first txty_(s)_twth, second xyxy(scaled in [0, 1])
                'conf': conf,
                'cls_prob': cls_prob
            }
            res[key] = now_size_res

        return res

    @staticmethod
    def split_target(
            target: dict,
            anchor_number_for_single_size: int,
    ):
        res = {}
        # key : 'C3', 'C4', 'C5'
        # val : dict --> {'position': xxx, 'conf': xxx, 'cls_prob': xxx}
        for key, x in target.items():
            N, C, H, W = x.shape
            K = C // anchor_number_for_single_size
            # K = (x, y, w, h, conf, kinds0, kinds1, ...)
            # C = anchor_number * K
            x = x.view(N, anchor_number_for_single_size, K, H, W)
            x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

            position = [None, x[..., 0:4]]  # scaled in [0, 1]
            conf = x[..., 4]  # N * H * W * a_n
            cls_prob = x[..., 5:]  # N * H * W * a_n * ...

            now_size_res = {
                'position': position,  # first txty_(s)_twth, second xyxy(scaled in [0, 1])
                'conf': conf,
                'cls_prob': cls_prob
            }
            res[key] = now_size_res

        return res

    @staticmethod
    def compute_anchor_response_result(
            anchor_pre_wh: dict,
            grid_number_dict: dict,
            abs_gt_pos: Union[tuple, list],
            image_wh: Union[tuple, list],
            iou_th: float = 0.6,
    ):
        keys = list(anchor_pre_wh.keys())
        values = np.array(list(anchor_pre_wh.values()))
        anchor_pre_wh = values.reshape(-1, 2).tolist()

        grid_number_ = np.array(list(grid_number_dict.values())).repeat(3, axis=1)
        grid_number_ = grid_number_.reshape(-1, 2).tolist()
        # ----------------------------------------------------------------------
        best_index = 0
        best_iou = 0
        weight_vec = []
        iou_vec = []
        gt_w = abs_gt_pos[2] - abs_gt_pos[0]
        gt_h = abs_gt_pos[3] - abs_gt_pos[1]

        if gt_w < 1e-4 or gt_h < 1e-4:
            # valid obj box
            return None

        s1 = gt_w * gt_h
        for index, val in enumerate(anchor_pre_wh):

            grid_number = grid_number_[index]

            anchor_w = val[0] / grid_number[0] * image_wh[0]   # scaled on image
            anchor_h = val[1] / grid_number[1] * image_wh[1]   # scaled on image

            s0 = anchor_w * anchor_h
            inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
            union = s0 + s1 - inter
            iou = inter / (union + 1e-8)
            if iou >= best_iou:
                best_index = index
                best_iou = iou
            weight_vec.append(
                2.0 - (gt_w / image_wh[0]) * (gt_h / image_wh[1])
            )
            iou_vec.append(iou)

        for iou_index in range(len(iou_vec)):
            if iou_index != best_index:
                if iou_vec[iou_index] >= iou_th:
                    weight_vec[iou_index] = - 1.0  # ignore this anchor
                else:
                    weight_vec[iou_index] = 0.0  # negative anchor
        # ----------------------------------------------------------------------
        w_v = np.array(weight_vec).reshape(values.shape[0], values.shape[1]).tolist()
        # (3, 3)
        res = {}
        for anchor_index, anchor_key in enumerate(keys):
            res[anchor_key] = w_v[anchor_index]

        return res

    @staticmethod
    def make_target(
            labels: list,
            anchor_pre_wh: dict,
            image_wh: tuple,
            grid_number: dict,
            kinds_name: list,
            iou_th: float = 0.6,
    ) -> dict:
        '''

        Args:
            labels: [label0, label1, ...]
                    label --> [obj0, obj1, ...]
                    obj --> [kind_name, x, y, x, y]  not scaled
            anchor_pre_wh:  key --> "for_s", "for_m", "for_l"
            image_wh:
            grid_number: key --> "for_s", "for_m", "for_l"
            kinds_name:
            iou_th:

        Returns:
            {
                "for_s": (N, a_n, 5+k_n, 52, 52) --> (N, -1, 52, 52)
                "for_m": (N, a_n, 5+k_n, 26, 26) --> (N, -1, 26, 26)
                "for_l": (N, a_n, 5+k_n, 13, 13) --> (N, -1, 13, 13)
            }

        '''
        kinds_number = len(kinds_name)
        N = len(labels)
        res = {}
        for anchor_key, val in grid_number.items():
            a_n, H, W = len(anchor_pre_wh[anchor_key]), val[1], val[0]
            res[anchor_key] = torch.zeros(size=(N, a_n, 5 + kinds_number, H, W))

        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = kinds_name.index(obj[0])
                abs_pos = obj[1:]

                weight_dict = YOLOV3Tool.compute_anchor_response_result(
                    anchor_pre_wh,
                    grid_number,
                    abs_pos,
                    image_wh,
                    iou_th
                )
                # weight_dict : key --> anchor_key, value --> weight of pre_anchor with gt_obj
                # (3, 3)
                if weight_dict is None:
                    continue

                pos = [val/image_wh[0] for val in abs_pos]  # scaled in [0, 1]

                for anchor_key in anchor_pre_wh.keys():
                    weight_vec = weight_dict[anchor_key]  # weight_vec of one anchor size
                    grid_size = (
                        image_wh[0] // grid_number[anchor_key][0],
                        image_wh[1] // grid_number[anchor_key][1]
                    )

                    grid_index = (
                        int((abs_pos[0] + abs_pos[2]) * 0.5 // grid_size[0]),  # w -- on x-axis
                        int((abs_pos[1] + abs_pos[3]) * 0.5 // grid_size[1])  # h -- on y-axis
                    )
                    for weight_index, weight_value in enumerate(weight_vec):
                        res[anchor_key][batch_index, weight_index, 4, grid_index[1], grid_index[0]] = weight_value
                        if weight_value != -1 and weight_value != 0:
                            res[anchor_key][batch_index, weight_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(
                                pos)
                            res[anchor_key][batch_index, weight_index, int(5 + kind_int), grid_index[1], grid_index[0]] = 1.0

        for anchor_key, val in grid_number.items():
            H, W = val[1], val[0]
            res[anchor_key] = res[anchor_key].view(N, -1, H, W)
        return res

    @staticmethod
    def xywh_to_xyxy(
            position: torch.Tensor,
            anchor_pre_wh_for_single_size: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        # -1 * H * W * a_n * 4
        N, _, _, a_n, _ = position.shape

        grid = YOLOV3Tool.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2

        grid_index = grid.to(position.device)

        pre_wh = torch.tensor(
            anchor_pre_wh_for_single_size,
            dtype=torch.float32
        )
        pre_wh = pre_wh.to(position.device)
        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (torch.sigmoid(a_b) + grid_index) / grid_number[0]  # scaled in [0, 1]
        w_h = torch.exp(m_n) * pre_wh.expand_as(m_n) / grid_number[0]  # scaled in [0, 1]

        x_y_0 = center_x_y - 0.5 * w_h
        # x_y_0[x_y_0 < 0] = 0
        x_y_1 = center_x_y + 0.5 * w_h
        # x_y_1[x_y_1 > grid_number] = grid_number
        res = torch.cat((x_y_0, x_y_1), dim=-1)
        return res.clamp_(0.0, 1.0)  # scaled in [0, 1]

    @staticmethod
    def xyxy_to_xy_s_wh(
            position: torch.Tensor,
            anchor_pre_wh_for_single_size: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        '''

        Args:
            position: scaled in [0, 1]
            anchor_pre_wh_for_single_size:
            grid_number:

        Returns:

        '''
        N, _, _, a_n, _ = position.shape

        grid = YOLOV3Tool.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2

        grid_index = grid.to(position.device)

        pre_wh = torch.tensor(
            anchor_pre_wh_for_single_size,
            dtype=torch.float32
        )
        pre_wh = pre_wh.to(position.device)

        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (a_b + m_n) * 0.5   # scaled in [0, 1]

        w_h = m_n - a_b  # scaled in [0, 1]

        txy_s = center_x_y * grid_number[0] - grid_index
        txy_s.clamp_(0.0, 1.0)  # be careful!!!, many center_x_y is zero !!!!

        twh = torch.log(w_h * grid_number[0] / pre_wh.expand_as(w_h) + 1e-20)

        return torch.cat((txy_s, twh), dim=-1)



