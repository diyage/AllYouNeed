"""*
估值部分准备先用工具包，后续自己写map计算
"""
from Package.Task.ObjectDetection.D2.Dev.evaluator import FinalEvaluator
from Package.Task.ObjectDetection.D2.YOLO.VX.Model import YOLOVXModel
from Package.Task.ObjectDetection.D2.YOLO.VX.Predictor import YOLOVXPredictor
from Package.Task.ObjectDetection.D2.YOLO.VX.Typing import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *
import torch


class YOLOVXEvaluator(FinalEvaluator):
    def __init__(
            self,
            model: YOLOVXModel,
            predictor: YOLOVXPredictor,
            cls_num: int,
    ):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device
        self.predictor = predictor
        self.cls_num = cls_num

    def convert_info_for_metrics(
            self,
            data_loader: DataLoader,
            desc: str = 'convert info for metrics',
    ):
        """
        这个函数的作用主用是将模型的解码结果（使用predictor）和样本的真实标签
        转化成适合计算metrics的格式
        格式大概是这样：
        [
            [img_ind, box_ind, cls_ind, x1, y1, x2, y2, score],
            ...
        ]
        """
        pre = []
        gt = []
        img_ind = 0
        gt_box_ind = 0
        pre_box_ind = 0

        for _, item in enumerate(tqdm(
            data_loader,
            desc=desc,
            position=0
        )):
            images: torch.Tensor = item[0]
            labels: List[LABEL] = item[1]

            self.model.eval()
            images = images.to(self.device)
            outputs = self.model(images)

            pre_decode: List[KPS_VEC] = self.predictor.post_process(outputs)

            for i in range(images.shape[0]):
                for obj in labels[i]:
                    gt.append(
                        [
                            img_ind, gt_box_ind, obj[0], *obj[1], 1.0
                        ]
                    )
                    gt_box_ind += 1

                for kps in pre_decode[i]:
                    pre.append(
                        [
                            img_ind, pre_box_ind, kps[0], *kps[1], kps[2]
                        ]
                    )
                    pre_box_ind += 1
                img_ind += 1

        return pre, gt
