import torch
from Package.Task.ObjectDetection.D2.Dev import DevTool


class YOLOV2Tool(DevTool):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_grid_number_and_pre_anchor_w_h(
            image_wh: tuple,
            image_shrink_rate: tuple,
            pre_anchor_w_h_rate: tuple
    ):

        grid_number = (
            image_wh[0] // image_shrink_rate[0],
            image_wh[1] // image_shrink_rate[1]
        )
        pre_anchor_w_h = tuple([
            (rate[0] * grid_number[0], rate[1] * grid_number[1]) for rate in pre_anchor_w_h_rate
        ])

        return grid_number, pre_anchor_w_h

    @staticmethod
    def make_target(
            labels: list,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th: float = 0.6,
    ) -> torch.Tensor:
        """
        Args:
            labels: [
                [obj, obj, obj, ...],               --> one image
                ...
            ]
                obj = [kind_name: str, x, y, x, y]  --> one obj
            anchor_pre_wh: [
                [w0, h0],
                [w1, h1],
                ...
            ]
            image_wh: [image_w, image_h]
            grid_number: [grid_w, grid_h]
            kinds_name: [kind_name0, kinds_name1, ... ]
            iou_th:

        Returns:
            (N, a_n * (5 + kinds_number), H, W)
        """

        kinds_number = len(kinds_name)
        N, a_n, H, W = len(labels), len(anchor_pre_wh), grid_number[1], grid_number[0]

        targets = torch.zeros(size=(N, a_n, 5 + kinds_number, H, W))

        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = kinds_name.index(obj[0])
                abs_pos = obj[1:]

                best_index, weight_vec = YOLOV2Tool.compute_anchor_response_result(
                    anchor_pre_wh,
                    abs_pos,
                    grid_number,
                    image_wh,
                    iou_th
                )
                if best_index == -1:
                    continue

                grid_size = (
                    image_wh[0] // grid_number[0],
                    image_wh[1] // grid_number[1]
                )

                grid_index = (
                    int((abs_pos[0] + abs_pos[2]) * 0.5 // grid_size[0]),  # w -- on x-axis
                    int((abs_pos[1] + abs_pos[3]) * 0.5 // grid_size[1])  # h -- on y-axis
                )
                pos = tuple(abs_pos)

                for weight_index, weight_value in enumerate(weight_vec):
                    targets[batch_index, weight_index, 4, grid_index[1], grid_index[0]] = weight_value
                    # conf / weight --->
                    # -1, ignore
                    # 0, negative
                    # >0 [1, 2], positive
                    if weight_index == best_index:
                        targets[batch_index, weight_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(
                            pos)
                        targets[batch_index, weight_index, int(5 + kind_int), grid_index[1], grid_index[0]] = 1.0

        return targets.view(N, -1, H, W)

    @staticmethod
    def compute_anchor_response_result(
            anchor_pre_wh: tuple,
            abs_gt_pos: tuple,
            grid_number: tuple,
            image_wh: tuple,
            iou_th: float = 0.6,
    ):

        best_index = 0
        best_iou = 0
        weight_vec = []
        iou_vec = []
        gt_w = abs_gt_pos[2] - abs_gt_pos[0]
        gt_h = abs_gt_pos[3] - abs_gt_pos[1]

        if gt_w < 1e-4 or gt_h < 1e-4:
            # valid obj box
            return -1, []

        s1 = gt_w * gt_h
        for index, val in enumerate(anchor_pre_wh):
            anchor_w = val[0] / grid_number[0] * image_wh[0]
            anchor_h = val[1] / grid_number[1] * image_wh[1]

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

        return best_index, weight_vec

    @staticmethod
    def split_target(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ):
        N, C, H, W = x.shape
        K = C // anchor_number  # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

        position = [None, x[..., 0:4]]

        conf = x[..., 4]  # N * H * W * a_n
        cls_prob = x[..., 5:]  # N * H * W * a_n * ...

        res = {
            'position': position,  # first txty_(s)_twth, second xyxy(not scaled)
            'conf': conf,
            'cls_prob': cls_prob
        }
        return res

    @staticmethod
    def split_predict(
            x: torch.Tensor,
            anchor_number,
            *args,
            **kwargs
    ):
        N, C, H, W = x.shape
        K = C // anchor_number  # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

        position = [x[..., 0:4], None]

        conf = x[..., 4]  # N * H * W * a_n
        cls_prob = x[..., 5:]  # N * H * W * a_n * ...

        res = {
            'position': position,  # first txty_(s)_twth, second xyxy(not scaled)
            'conf': conf,
            'cls_prob': cls_prob
        }
        return res

    @staticmethod
    def xywh_to_xyxy(
            position: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:
        # -1 * H * W * a_n * 4
        N, _, _, a_n, _ = position.shape

        grid = YOLOV2Tool.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2

        grid_index = grid.to(position.device)

        pre_wh = torch.tensor(
            anchor_pre_wh,
            dtype=torch.float32
        )
        pre_wh = pre_wh.to(position.device)
        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (torch.sigmoid(a_b) + grid_index) / grid_number[0] * image_wh[0]
        w_h = torch.exp(m_n) * pre_wh.expand_as(m_n) / grid_number[0] * image_wh[0]

        x_y_0 = center_x_y - 0.5 * w_h
        # x_y_0[x_y_0 < 0] = 0
        x_y_1 = center_x_y + 0.5 * w_h
        # x_y_1[x_y_1 > grid_number] = grid_number
        res = torch.cat((x_y_0, x_y_1), dim=-1)
        return res.clamp_(0.0, image_wh[0]-1)

    @staticmethod
    def xyxy_to_xywh(
            position: torch.Tensor,
            anchor_pre_wh: tuple,
            image_wh: tuple,
            grid_number: tuple,
    ) -> torch.Tensor:

        # def arc_sigmoid(x: torch.Tensor) -> torch.Tensor:
        #     return - torch.log(1.0 / (x + 1e-8) - 1.0)

        # -1 * H * W * a_n * 4
        N, _, _, a_n, _ = position.shape

        grid = YOLOV2Tool.get_grid(grid_number)
        # H * W * 2

        grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                                      grid_number[0],
                                                      grid_number[1],
                                                      a_n,
                                                      2)

        # -1 * H * W * a_n * 2

        grid_index = grid.to(position.device)

        pre_wh = torch.tensor(
            anchor_pre_wh,
            dtype=torch.float32
        )
        pre_wh = pre_wh.to(position.device)

        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = (a_b + m_n) * 0.5

        w_h = m_n - a_b

        # txy = arc_sigmoid(center_x_y / image_wh[0] * grid_number[0] - grid_index)
        txy_s = center_x_y / image_wh[0] * grid_number[0] - grid_index
        # center_xy = (sigmoid(txy) + grid_index) / grid_number * image_wh
        # we define txy_s = sigmoid(txy)
        # be careful ,we do not use arc_sigmoid method
        # if you use txy(in model output), please make sure (use sigmoid)
        txy_s.clamp_(0.0, 1.0)  # be careful!!!, many center_x_y is zero !!!!
        twh = torch.log(w_h/image_wh[0]*grid_number[0]/pre_wh.expand_as(w_h) + 1e-20)

        return torch.cat((txy_s, twh), dim=-1)
