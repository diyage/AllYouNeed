from typing import *
import numpy as np


class YOLOVXDataSetConfig:
    root_path: str = '/home/dell/data/DataSet/COCO/'
    years: list = ['2014', '2017']

    # data set root dir
    image_size: int = 640

    # width_height_center: list = [
    #     [80.93151919, 115.35980124],
    #     [218.9952542, 360.14068262],
    #     [511.91991605, 480.48462577]
    # ]  # this is used for voc image_size:640

    width_height_center: list = [
        [49.31695546, 64.27518682],
        [173.54360828, 274.19121854],
        [471.14308384, 444.86262709],
    ]  # this is used for coco image_size:640

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
    batch_size = 16
    lr: float = 1e-3
    reach_base_lr_cost_epoch: int = 5

    use_mosaic: bool = False
    use_mix_up: bool = False
    mix_up_lambda: float = 0.5

    weight_position: float = 5.0
    weight_conf_has_obj: float = 1.0
    weight_conf_no_obj: float = 1.0
    weight_conf_obj: float = 1.0
    weight_cls: float = 1.0


class YOLOVXEvalConfig:
    eval_frequency = 5
    conf_th_for_eval: float = 0.0
    cls_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001
    iou_th_for_eval: float = 0.5  # used for nms
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

    def init_lr_mapping(
            self,
            max_epoch: int,
            max_batch: int,
            reach_base_lr_cost_epoch: int,
            base_lr: float,
    ):
        lr_mapping = {
            epoch: {} for epoch in range(max_epoch)
        }
        lr = np.linspace(
            0,
            base_lr,
            num=max_batch * reach_base_lr_cost_epoch
        ).tolist() + np.linspace(
            base_lr,
            0,
            num=max_batch * (max_epoch - reach_base_lr_cost_epoch)
        ).tolist()
        for epoch in range(max_epoch):
            for batch in range(max_batch):
                lr_mapping[epoch][batch] = lr[batch + epoch * max_batch]

        self.set_lr_mapping(
            lr_mapping
        )

    def init_mile_stone_mapping(
            self,
            max_epoch: int,
            max_batch: int,
            reach_base_lr_cost_epoch: int,
            base_lr: float,
            mile_stone: list,
            alpha: float = 0.1
    ):
        lr_mapping = {
            epoch: {} for epoch in range(max_epoch)
        }

        lr = np.linspace(
            0,
            base_lr,
            num=max_batch * reach_base_lr_cost_epoch
        ).tolist()

        now_lr = base_lr

        for epoch in range(reach_base_lr_cost_epoch, max_epoch):
            for batch in range(max_batch):

                if epoch in mile_stone and batch == 0:
                    now_lr *= alpha
                lr.append(now_lr)

        for epoch in range(max_epoch):
            for batch in range(max_batch):
                lr_mapping[epoch][batch] = lr[batch + epoch * max_batch]
                # print("{}, {}: {:.7f}".format(epoch, batch, lr_mapping[epoch][batch]))

        self.set_lr_mapping(
            lr_mapping
        )

    def set_lr_mapping(
            self,
            new_lr_mapping: Dict[int, Dict[int, float]]
    ):
        self.lr_mapping = new_lr_mapping

