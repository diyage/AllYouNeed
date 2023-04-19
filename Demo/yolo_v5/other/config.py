
import numpy as np


class YOLOV5DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/COCO/'
    years: list = ['2014', '2017']

    # data set root dir
    image_size: tuple = (608, 608)
    image_shrink_rate: dict = {
        'for_s': (8, 8),
        'for_m': (16, 16),
        'for_l': (32, 32),
    }
    pre_anchor_w_h_rate: dict = {
        'for_s': ((0.07846153846153846, 0.11461538461538462),
                  (0.12076923076923077, 0.26),
                  (0.3046153846153846, 0.23153846153846153)),

        'for_m': ((0.18846153846153849, 0.48538461538461536),
                  (0.4284615384615385, 0.42923076923076925),
                  (0.31153846153846154, 0.7084615384615385)),

        'for_l': ((0.7976923076923076, 0.4676923076923077),
                  (0.5476923076923077, 0.783076923076923),
                  (0.8784615384615384, 0.8623076923076923)),
    }
    single_an: int = 3
    kinds_name: list = ['person', 'bicycle', 'car', 'motorcycle',
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


class YOLOV5TrainConfig:
    max_epoch_on_detector = 500
    num_workers: int = 8
    device: str = 'cuda:0'
    batch_size = 32
    lr: float = 1e-3
    reach_base_lr_cost_epoch: int = 2

    use_mosaic: bool = True
    # use_mixup: bool = True

    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0


class YOLOV5EvalConfig:
    eval_frequency = 10
    conf_th_for_eval: float = 0.0
    prob_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001
    iou_th_for_eval: float = 0.5
    use_07_metric: bool = False


class YOLOV5VisualizationConfig:

    conf_th_for_show: float = 0.0
    prob_th_for_show: float = 0.0
    score_th_for_show: float = 0.3

    iou_th_for_show: float = 0.5


class YOLOV5Config:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    iou_th_for_make_target: float = 0.5
    multi_gt: bool = True
    data_config = YOLOV5DataSetConfig()
    train_config = YOLOV5TrainConfig()
    eval_config = YOLOV5EvalConfig()
    show_config = YOLOV5VisualizationConfig()


