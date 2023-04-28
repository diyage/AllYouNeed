from typing import *
import numpy as np


class YOLOVXDataSetConfig:
    root_path: str = '/home/dell/data/DataSet/COCO/'
    years: list = ['2014', '2017']

    # data set root dir
    image_size: int = 640
    image_shrink_rate: Tuple[int, int, int] = (8, 16, 32)
    use_label_type: bool = True

    kinds_name: List[str] = [
        'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(kinds_name))
    ]

    mean = [0.406, 0.456, 0.485]

    std = [0.225, 0.224, 0.229]


class YOLOVXTrainConfig:
    max_epoch_on_detector = 500
    num_workers: int = 8
    device: str = 'cuda:0'
    batch_size = 32
    lr: float = 1e-3
    reach_base_lr_cost_epoch: int = 2

    use_mosaic: bool = False
    # use_mixup: bool = True

    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls: float = 1.0


class YOLOVXEvalConfig:
    eval_frequency = 10
    conf_th_for_eval: float = 0.0
    cls_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001
    iou_th_for_eval: float = 0.5
    use_07_metric: bool = False


class YOLOVXVisualizationConfig:

    conf_th_for_show: float = 0.0
    cls_th_for_show: float = 0.0
    score_th_for_show: float = 0.3

    iou_th_for_show: float = 0.5


class YOLOVXConfig:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''

    multi_positives: bool = True
    data_config = YOLOVXDataSetConfig()
    train_config = YOLOVXTrainConfig()
    eval_config = YOLOVXEvalConfig()
    show_config = YOLOVXVisualizationConfig()
    lr_mapping: Dict[int, Dict[int, float]] = {}

    def set_lr_mapping(
            self,
            new_lr_mapping: Dict[int, Dict[int, float]]
    ):
        self.lr_mapping = new_lr_mapping

