
import numpy as np


class YOLOV4DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
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
    kinds_name: list = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(kinds_name))
    ]

    mean = [0.406, 0.456, 0.485]

    std = [0.225, 0.224, 0.229]


class YOLOV4TrainConfig:
    max_epoch_on_detector = 200
    num_workers: int = 4
    device: str = 'cuda:0'
    batch_size = 16
    lr: float = 1e-3
    warm_up_end_epoch: int = 5
    use_mosaic: bool = True
    use_mixup: bool = True

    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0


class YOLOV4EvalConfig:
    eval_frequency = 10
    conf_th_for_eval: float = 0.0
    prob_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001
    iou_th_for_eval: float = 0.5
    use_07_metric: bool = False


class YOLOV4VisualizationConfig:

    conf_th_for_show: float = 0.0
    prob_th_for_show: float = 0.0
    score_th_for_show: float = 0.3

    iou_th_for_show: float = 0.5


class YOLOV4Config:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    iou_th_for_make_target: float = 0.5
    multi_gt: bool = True
    data_config = YOLOV4DataSetConfig()
    train_config = YOLOV4TrainConfig()
    eval_config = YOLOV4EvalConfig()
    show_config = YOLOV4VisualizationConfig()


