from .Predictor import YOLOV4Predictor
from .Tools import YOLOV4Tool
from .Model import YOLOV4Model
from Package.Task.ObjectDetection.D2.Dev import DevEvaluator


class YOLOV4Evaluator(DevEvaluator):
    def __init__(
            self,
            model: YOLOV4Model,
            predictor: YOLOV4Predictor,
            iou_th_for_make_target: float,
            multi_gt: bool,
    ):
        super().__init__(
            model,
            predictor,
            iou_th_for_make_target
        )

        self.predictor = predictor
        self.anchor_keys = self.predictor.anchor_keys
        self.multi_gt = multi_gt

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV4Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels
    ):
        targets = YOLOV4Tool.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
            multi_gt=self.multi_gt
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets
