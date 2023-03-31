from .Predictor import YOLOV5Predictor
from .Tools import YOLOV5Tool
from .Model import YOLOV5Model
from Package.Task.ObjectDetection.D2.Dev import DevVisualizer


class YOLOV5Visualizer(DevVisualizer):
    def __init__(
            self,
            model: YOLOV5Model,
            predictor: YOLOV5Predictor,
            class_colors: list,
            iou_th_for_make_target: float,
            multi_gt: bool
    ):
        super().__init__(
            model,
            predictor,
            class_colors,
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
        self.grid_number, self.pre_anchor_w_h = YOLOV5Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels,
    ):
        targets = YOLOV5Tool.make_target(
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

