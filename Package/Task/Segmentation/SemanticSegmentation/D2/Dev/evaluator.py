"""
Used for evaluating some metrics of detector.
"""
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
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
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.predictor = predictor

    def make_targets(
            self,
            labels: List[np.ndarray]
    ):
        targets = DevTool.make_target(
            labels
        )
        return targets.to(self.device)

    def eval_semantic_segmentation_accuracy(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval semantic segmentation accuracy',
    ):
        acc_vec_include_background = []
        acc_vec = []
        for batch_id, (images, objects_vec, masks_vec) in enumerate(tqdm(data_loader_test,
                                                                         desc=desc,
                                                                         position=0)):
            self.model.eval()
            images = images.to(self.device)

            targets = self.make_targets(masks_vec)

            output = self.model(images)

            gt_decode = self.predictor.decode_target(targets)  # type: np.ndarray
            pre_decode = self.predictor.decode_predict(output)  # type: np.ndarray
            """
            n * h * w * c(or mask_num or kinds_num + 1)
            """
            pre_mask_vec = np.argmax(pre_decode, axis=-1)
            gt_mask_vec = np.argmax(gt_decode, axis=-1)

            acc = np.mean((pre_mask_vec == gt_mask_vec).astype(np.float32))
            acc_vec_include_background.append(acc)
            """
                do not consider background, it will cause very high accuracy !!
            """
            except_background = gt_mask_vec != 0
            acc = np.mean((pre_mask_vec[except_background] == gt_mask_vec[except_background]).astype(np.float32))
            acc_vec.append(acc)

        print(
            '\nsemantic segmentation accuracy:{:.2%}(just consider positive pixels), {:.2%}(include background)'.format(
                np.mean(acc_vec),
                np.mean(acc_vec_include_background)
            ))
