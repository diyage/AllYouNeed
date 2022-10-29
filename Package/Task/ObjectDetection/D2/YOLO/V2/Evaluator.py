from .Predictor import YOLOV2Predictor
from .Tools import YOLOV2Tool
from .Model import YOLOV2Model
from Package.Task.ObjectDetection.D2.Dev import DevEvaluator


class YOLOV2Evaluator(DevEvaluator):
    def __init__(
            self,
            model: YOLOV2Model,
            predictor: YOLOV2Predictor,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            predictor,
            iou_th_for_make_target
        )

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV2Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels
    ):
        return YOLOV2Tool.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            iou_th=self.iou_th_for_make_target,
        ).to(self.device)
