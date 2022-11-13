from ..Dev import DevTool
import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import typing


class MTCNNTool(DevTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    def make_target(
            image_type: int,
            key_point: torch.Tensor,
            positive_position_offset: torch.Tensor,
            part_position_offset: torch.Tensor,
            *args,
            **kwargs
    ) -> dict:
        assert image_type in [0, 1, 2, 3, 4, 5]
        # please see MTCNNDataSet
        # 0 used for computing key-point loss
        # 1 used for computing positive loss
        # 2, 3, 4 used for computing negative loss
        # 5 used for computing part loss
        batch_size = key_point.shape[0]
        res = {
            'cls': None,
            'pos_offset': None,
            'key_point': None,
        }
        if image_type == 0:
            # 0 used for computing key-point loss

            res['key_point'] = key_point

            return res
        elif image_type == 1:
            # 1 used for computing positive loss
            # classification and box regression

            res['cls'] = torch.ones(
                size=(batch_size, ),
                dtype=torch.long
            )
            res['pos_offset'] = positive_position_offset
            return res

        elif image_type == 5:
            # 5 used for computing part loss
            # just used for box regression
            res['pos_offset'] = part_position_offset
            return res
        else:
            # 2, 3, 4 used for computing negative loss
            # just used for classification

            res['cls'] = torch.zeros(
                size=(batch_size, ),
                dtype=torch.long
            )

            return res

    @staticmethod
    def split_target(
            *args,
            **kwargs
    ):
        raise RuntimeError

    @staticmethod
    def split_net_predict(
            predict: dict,
            net_type: str,
    ) -> dict:
        if net_type == 'p':
            pre_cls: torch.Tensor = predict.get('cls')  # (1, 2, m, n)
            pre_cls = pre_cls.permute(0, 2, 3, 1)  # (1, m, n, 2)
            pre_cls = pre_cls.contiguous().view(-1, 2)

            pre_position_offset: torch.Tensor = predict.get('pos_offset')  # (1, 4, m, n)
            pre_position_offset = pre_position_offset.permute(0, 2, 3, 1)  # (1, m, n, 4)
            pre_position_offset = pre_position_offset.contiguous().view(-1, 4)

            pre_key_point_offset: torch.Tensor = predict.get('key_point')  # (1, 10, m, n)
            pre_key_point_offset = pre_key_point_offset.permute(0, 2, 3, 1)  # (1, m, n, 10)
            pre_key_point_offset = pre_key_point_offset.contiguous().view(-1, 10)
            return {
                'cls': pre_cls,
                'pos_offset': pre_position_offset,
                'key_point': pre_key_point_offset
            }
        else:
            return predict

    @staticmethod
    def split_predict(
            *args,
            **kwargs
    ):
        raise RuntimeError

    @staticmethod
    def get_grid(
            grid_number: tuple
    ):
        index_h = torch.tensor(list(range(grid_number[0])), dtype=torch.float32)
        index_w = torch.tensor(list(range(grid_number[1])), dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(index_h, index_w)
        grid = torch.cat((grid_c.unsqueeze(-1), grid_r.unsqueeze(-1)), dim=-1)
        # H * W * 2
        return grid

    @staticmethod
    def generate_candidate_box(
            original_h: int,
            original_w: int,
            mapping_h: int,
            mapping_w: int,
            box_size: int
    ) -> torch.Tensor:
        grid = MTCNNTool.get_grid((mapping_h, mapping_w))  # h * w * 2   and  2:  (x, y)

        """
        rate_for_x * mapping_w = original_w
        """
        rate_for_y = 1.0 * original_h / mapping_h
        rate_for_x = 1.0 * original_w / mapping_w
        x1 = grid[..., 0] * rate_for_x
        y1 = grid[..., 1] * rate_for_y
        x2 = x1 + box_size
        y2 = y1 + box_size
        return torch.stack(
            [x1, y1, x2, y2],
            dim=-1
        )

    @staticmethod
    def adjust_and_filter_candidate_box_and_landmark(
            candidate_box: torch.Tensor,
            pre_cls: torch.Tensor,
            pre_position_offset: torch.Tensor,
            pre_key_point_offset: torch.Tensor,
            score_threshold: float,
            nms_threshold: float
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        score, pre_kind = pre_cls.softmax(dim=-1).max(dim=-1)

        # filter negative boxes
        is_face = (pre_kind == 1)
        mask = is_face.float() * (score > score_threshold).float()
        mask_bool = mask.bool()

        candidate_box = candidate_box[mask_bool]
        score = score[mask_bool]
        pre_position_offset = pre_position_offset[mask_bool]
        pre_key_point_offset = pre_key_point_offset[mask_bool]

        # adjust box and landmark
        x1, y1, x2, y2 = candidate_box[:, 0], candidate_box[:, 1], candidate_box[:, 2], candidate_box[:, 3]
        candidate_box_x1_y1 = torch.stack([x1]*5 + [y1]*5, dim=-1)  # (-1, 10)
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        size_for_position = torch.stack([w, h, w, h], dim=-1)  # (-1, 4)
        size_for_landmark = torch.stack([w]*5 + [h]*5, dim=-1)  # (-1, 10)

        adjusted_box = pre_position_offset * size_for_position + candidate_box
        adjusted_landmark = pre_key_point_offset * size_for_landmark + candidate_box_x1_y1

        # filter negative boxes
        mask_0 = (adjusted_box[:, 0] > 0).float() * (adjusted_box[:, 1] > 0).float() \
            * (adjusted_box[:, 2] > 0).float() * (adjusted_box[:, 3] > 0).float()

        mask_1 = (adjusted_box[:, 0] < adjusted_box[:, 2]).float() * (adjusted_box[:, 1] < adjusted_box[:, 3]).float()

        mask_bool = (mask_0 * mask_1).bool()

        adjusted_box = adjusted_box[mask_bool]
        score = score[mask_bool]
        adjusted_landmark = adjusted_landmark[mask_bool]

        # nms
        keep = MTCNNTool.single_kind_nms(
            adjusted_box,
            score,
            nms_threshold
        )
        return adjusted_box[keep], adjusted_landmark[keep]

    @staticmethod
    def get_more_scale_size_image(
            img: np.ndarray,
            image_scale_rate: float,
            min_size: int,
    ):

        res = []
        now_scale_rate = 1.0
        now_h, now_w, _ = img.shape
        original_h, original_w = now_h, now_w

        while now_w >= min_size and now_h >= min_size:
            res.append([
                img,
                1.0 * now_h / original_h,
                1.0 * now_w / original_w
            ])
            now_scale_rate *= image_scale_rate
            now_h = int(1.0 * original_h * now_scale_rate)
            now_w = int(1.0 * original_w * now_scale_rate)
            img: np.ndarray = cv2.resize(img, (now_w, now_h))

        return res

    @staticmethod
    def get_cropped_images_with_candidate_box(
            img: np.ndarray,
            candidate_box: np.ndarray,
            cropped_size: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        h, w, _ = img.shape
        res_cropped_images = []
        res_candidate_box = []

        candidate_box = candidate_box.astype(np.int32)
        for box in candidate_box:
            x1, y1, x2, y2 = box

            m, n = y2 - y1 + 1, x2 - x1 + 1
            if m < n:
                y2 = y2 + n - m
                if y2 > h:
                    continue
            else:
                x2 = x2 + m - n
                if x2 > w:
                    continue
            if y2 <= y1 or x2 <= x1:
                continue

            cropped_img = img[y1:y2, x1:x2, :]

            cropped_img = cv2.resize(cropped_img, (cropped_size, cropped_size))
            res_cropped_images.append(cropped_img)
            res_candidate_box.append([x1, y1, x2, y2])
        return np.array(res_cropped_images, dtype=np.uint8), np.array(res_candidate_box, dtype=np.float32)
