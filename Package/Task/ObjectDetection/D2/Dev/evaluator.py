"""
Used for evaluating some metrics of detector.
"""
from abc import abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from typing import Union
import numpy as np
from .model import DevModel
from .tools import DevTool
from .predictor import DevPredictor
from Package.BaseDev import BaseEvaluator


class DevEvaluator(BaseEvaluator):
    def __init__(
            self,
            model: DevModel,
            predictor: DevPredictor,
            iou_th_for_make_target: float
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.predictor = predictor

        self.kinds_name = predictor.kinds_name
        self.iou_th_for_eval = self.predictor.iou_th

        self.pre_anchor_w_h_rate = self.predictor.pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = self.predictor.image_shrink_rate
        self.grid_number = None

        self.image_size = None  # type: tuple
        self.change_image_wh(self.predictor.image_size)

        self.iou_th_for_make_target = iou_th_for_make_target

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ):
        pass

    #################################################################################
    """
    for eval map
    """
    @staticmethod
    def iou_score(bboxes_a, bboxes_b):
        """
            bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
            bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
        """
        tl = torch.max(bboxes_a[..., :2], bboxes_b[..., :2])
        br = torch.min(bboxes_a[..., 2:], bboxes_b[..., 2:])
        area_a = torch.prod(bboxes_a[..., 2:] - bboxes_a[..., :2], dim=-1)
        area_b = torch.prod(bboxes_b[..., 2:] - bboxes_b[..., :2], dim=-1)

        en = (tl < br).type(tl.type()).prod(dim=-1)
        area_i = torch.prod(br - tl, dim=-1) * en  # * ((tl < br).all())

        area_union = area_a + area_b - area_i
        iou = area_i / (area_union + 1e-20)

        return iou

    @staticmethod
    def compute_iou(
            boxes0: Union[torch.Tensor, np.ndarray, list],
            boxes1: Union[torch.Tensor, np.ndarray, list]
    ) -> torch.Tensor:
        if isinstance(boxes0, np.ndarray) or isinstance(boxes0, list):
            boxes0 = torch.tensor(boxes0)
            boxes1 = torch.tensor(boxes1)
            if len(boxes0.shape) == 1:
                boxes0 = boxes0.unsqueeze(0)
                boxes1 = boxes1.unsqueeze(0)

        return DevEvaluator.iou_score(boxes0, boxes1)

    @staticmethod
    def get_pre_kind_name_tp_score_and_gt_num(
            pre_kind_name_pos_score: list,
            gt_kind_name_pos_score: list,
            kinds_name: list,
            iou_th: float = 0.5
    ):
        """
        just used for one image(all predicted box information(s))
        Args:
            pre_kind_name_pos_score: [kps0, kps1, ...]   kps --> (kind_name, (x, y, x, y), score)
            gt_kind_name_pos_score:
            kinds_name: [kind_name0, kind_name1, ...]
            iou_th:

        Returns:
            (
                kind_tp_and_score,
                gt_num
            )
            kind_tp_and_score = [
                            [kind_name, is_tp(0.0/1.0), score],
                            ...
                            ]
            gt_num --> dict key is each kind_name, val is real gt_num(TP+FN) of this kind_name

        """
        pre_kind_name_pos_score = sorted(
            pre_kind_name_pos_score,
            key=lambda s: s[2],
            reverse=True
        )
        # sorted score from big to small
        kind_tp_and_score = []
        gt_num = {
            key: 0 for key in kinds_name
        }

        gt_has_used = []
        for gt_ in gt_kind_name_pos_score:
            gt_kind_name, _, _ = gt_
            gt_num[gt_kind_name] += 1
            gt_has_used.append(False)

        for pre_ in pre_kind_name_pos_score:
            pre_kind_name, pre_pos, pre_score = pre_
            is_tp = 0  # second element represents it tp(or fp)
            for gt_index, gt_ in enumerate(gt_kind_name_pos_score):
                gt_kind_name, gt_pos, gt_score = gt_
                if gt_kind_name == pre_kind_name and not gt_has_used[gt_index]:
                    iou = DevEvaluator.compute_iou(
                        list(pre_pos),
                        list(gt_pos)
                    )
                    if iou[0].item() > iou_th:
                        gt_has_used[gt_index] = True
                        is_tp = 1
                        break
                        # be careful

            kind_tp_and_score.append(
                [pre_kind_name, is_tp, pre_score]
            )
        return kind_tp_and_score, gt_num

    @staticmethod
    def calculate_pr(gt_num, tp_list, confidence_score):
        """
        calculate all p-r pairs among different score_thresh for one class, using `tp_list` and `confidence_score`.

        Args:
            gt_num (Integer): 某张图片中某类别的gt数量
            tp_list (List): 记录某张图片中某类别的预测框是否为tp的情况
            confidence_score (List): 记录某张图片中某类别的预测框的score值 (与tp_list相对应)

        Returns:
            recall
            precision

        """
        if gt_num == 0:
            return [0], [0]
        if isinstance(tp_list, (tuple, list)):
            tp_list = np.array(tp_list)
        if isinstance(confidence_score, (tuple, list)):
            confidence_score = np.array(confidence_score)

        assert len(tp_list) == len(confidence_score), "len(tp_list) and len(confidence_score) should be same"

        if len(tp_list) == 0:
            return [0], [0]

        sort_mask = np.argsort(-confidence_score)
        tp_list = tp_list[sort_mask]
        recall = np.cumsum(tp_list) / gt_num
        precision = np.cumsum(tp_list) / (np.arange(len(tp_list)) + 1)

        return recall.tolist(), precision.tolist()

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        """Compute VOC AP given precision and recall. If use_07_metric is true, uses
        the VOC 07 11-point method (default:False).
        """
        if isinstance(rec, (tuple, list)):
            rec = np.array(rec)
        if isinstance(prec, (tuple, list)):
            prec = np.array(prec)
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    # def eval_map(
    #         self,
    #         data_loader_test: DataLoader,
    #         desc: str = 'eval detector mAP',
    # ):
    #     # compute mAP
    #
    #     record = {
    #         key: [[], [], 0] for key in self.kinds_name
    #         # kind_name: [tp_list, score_list, gt_num]
    #     }
    #
    #     for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
    #                                                      desc=desc,
    #                                                      position=0)):
    #         self.model.eval()
    #         images = images.to(self.device)
    #
    #         targets = self.make_targets(labels)
    #         output = self.model(images)
    #
    #         gt_decode = self.predictor.decode_target(targets)  # kps_vec_s
    #         pre_decode = self.predictor.decode_predict(output)  # kps_vec_s
    #
    #         for image_index in range(images.shape[0]):
    #
    #             res = self.get_pre_kind_name_tp_score_and_gt_num(
    #                 pre_decode[image_index],
    #                 gt_decode[image_index],
    #                 kinds_name=self.kinds_name,
    #                 iou_th=self.iou_th_for_eval
    #             )
    #
    #             for pre_kind_name, is_tp, pre_score in res[0]:
    #                 record[pre_kind_name][0].append(is_tp)  # tp list
    #                 record[pre_kind_name][1].append(pre_score)  # score list
    #
    #             for kind_name, gt_num in res[1].items():
    #                 record[kind_name][2] += gt_num
    #
    #     # end for dataloader
    #
    #     ap_vec = []
    #     for kind_name in self.kinds_name:
    #         tp_list, score_list, gt_num = record[kind_name]
    #         recall, precision = self.calculate_pr(gt_num, tp_list, score_list)
    #         kind_name_ap = self.voc_ap(recall, precision)
    #         ap_vec.append(kind_name_ap)
    #
    #     mAP = np.mean(ap_vec)
    #     print('\nmAP:{:.2%}'.format(mAP))

    def get_info(
            self,
            decode: list,
            batch_id: int,
            batch_size: int
    ) -> list:
        info_vec = []
        for idx in range(len(decode)):
            image_id = batch_id * batch_size + idx
            for kps in decode[idx]:
                info = [
                    self.kinds_name.index(kps[0]),
                    *kps[1],
                    kps[2],
                    image_id
                ]
                info_vec.append(info)
        return info_vec

    def compute_ap(
            self,
            now_cls_pre_info_vec: np.ndarray,
            now_cls_gt_info_vec: np.ndarray,
            use_07_metric: bool = True
    ) -> float:

        tp_list = []
        score_list = []
        gt_num = now_cls_gt_info_vec.shape[0]

        gt_box_has_used = [False for _ in range(gt_num)]
        gt_box_has_used = np.array(gt_box_has_used)

        """
        pre-process
        """
        score_sorted_ind = np.argsort(now_cls_pre_info_vec[:, 5])[::-1]
        now_cls_pre_info_vec = now_cls_pre_info_vec[score_sorted_ind]
        """
        we expand now_cls_gt_info_vec
        dim 7: is the gt_num_ind ,
        could search gt_box_has_used by dim 7
        """
        tmp = [ind for ind in range(gt_num)]
        tmp = np.reshape(
            np.array(tmp, dtype=np.float32),
            newshape=(gt_num, 1)
        )

        now_cls_gt_info_vec = np.concatenate(
            (now_cls_gt_info_vec, tmp),
            axis=1
        )

        for info in now_cls_pre_info_vec:
            image_id = info[6]
            mask_0: np.ndarray = (gt_box_has_used == False)
            mask_1: np.ndarray = (now_cls_gt_info_vec[:, 6] == image_id)
            mask_float: np.ndarray = mask_0.astype(np.float32) * mask_1.astype(np.float32)

            score_list.append(info[5])
            if mask_float.sum() == 0:
                is_tp = 0
            else:
                mask = mask_float.astype(np.bool8)
                response_gt_info_vec = now_cls_gt_info_vec[mask]  # n * 8

                box_a = info[1:5]  # 4,
                boxs_b = response_gt_info_vec[:, 1:5]  # n * 4
                iou: torch.Tensor = self.compute_iou(
                    np.expand_dims(box_a, 0),
                    boxs_b
                )  # 1* n
                iou = np.reshape(
                    iou.numpy(),
                    newshape=(-1,)
                )
                iou_max_ind = np.argmax(iou)
                if iou[iou_max_ind] >= self.iou_th_for_eval:
                    gt_num_ind = int(response_gt_info_vec[iou_max_ind][-1])
                    gt_box_has_used[gt_num_ind] = True
                    is_tp = 1
                else:
                    is_tp = 0

            tp_list.append(is_tp)

        recall, precision = self.calculate_pr(
            gt_num,
            tp_list,
            score_list
        )
        ap: float = self.voc_ap(
            recall,
            precision,
            use_07_metric
        )
        return ap

    def eval_map(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval detector mAP',
            use_07_metric: bool = True
    ):
        print('')
        print('start eval mAP(use_07_metric: {})...'.format(use_07_metric).center(50, '*'))
        print('be careful, we do not ignore the difficult objects, so mAP may decrease a little...')

        pre_info_vec = []
        gt_info_vec = []
        """
        info_vec = [info_0, info_1, ...]
        info = [
            class_id,       # 0
            abs_pos,        # 1-4 (x y x y)  
            score,          # 5
            image_id,       # 6
        ]
        """
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.model.eval()
            images = images.to(self.device)

            targets = self.make_targets(labels)
            output = self.model(images)

            gt_decode = self.predictor.decode_target(targets)  # kps_vec_s
            pre_decode = self.predictor.decode_predict(output)  # kps_vec_s

            gt_info_vec += self.get_info(
                gt_decode,
                batch_id,
                data_loader_test.batch_size
            )
            pre_info_vec += self.get_info(
                pre_decode,
                batch_id,
                data_loader_test.batch_size
            )

        pre_info_vec = np.array(pre_info_vec, dtype=np.float32)
        gt_info_vec = np.array(gt_info_vec, dtype=np.float32)
        ap_vec = []
        for kind_ind in range(len(self.kinds_name)):
            ap = self.compute_ap(
                pre_info_vec[pre_info_vec[:, 0] == kind_ind],
                gt_info_vec[gt_info_vec[:, 0] == kind_ind],
                use_07_metric=use_07_metric
            )
            ap_vec.append(ap)
        print('\nmAP:{:.2%}'.format(np.mean(ap_vec)))

    #######################################################################################
