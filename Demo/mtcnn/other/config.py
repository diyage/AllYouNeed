class MTCNNDataSetConfig:
    CelebA_root: str = '/home/dell/data/DataSet/CelebA'
    CelebA_cache: str = 'cache_CelebA'
    WiderFace_root: str = '/home/dell/data/DataSet/WiderFace'
    WiderFace_cache: str = 'cache_Wider_Face'
    images_cropped_size: dict = {
        'p': (12, 12),
        'r': (24, 24),
        'o': (48, 48)
    }
    mean: list = [0.5, 0.5, 0.5]
    std: list = [0.5, 0.5, 0.5]


class MTCNNTrainConfig:
    batch_size: int = 256
    num_workers: int = 8
    lr: dict = {
        'p': 1e-4,
        'r': 1e-4,
        'o': 1e-4,
    }
    max_epoch_for_train: dict = {
        'p': 50,
        'r': 50,
        'o': 50,
    }

    device: str = 'cpu'

    cls_factor: dict = {
        'p': 1.0,
        'r': 1.0,
        'o': 1.0
    }
    box_factor: dict = {
        'p': 0.5,
        'r': 0.5,
        'o': 5.0
    }
    landmark_factor: dict = {
        'p': 0.5,
        'r': 0.5,
        'o': 50.0
    }
    stage_one_threshold: dict = {
        'cls': 0.5,
        'nms': 0.7
    }
    stage_two_threshold: dict = {
        'cls': 0.5,
        'nms': 0.7
    }


class MTCNNEvalConfig:
    eval_frequency: int = 1
    image_scale_rate: float = 0.7

    stage_one_threshold: dict = {
        'cls': 0.6,
        'nms': 0.7
    }
    stage_two_threshold: dict = {
        'cls': 0.7,
        'nms': 0.7
    }
    stage_three_threshold: dict = {
        'cls': 0.7,
        'nms': 0.3  # ???
    }


class MTCNNVisualizationConfig:
    pass


class MTCNNConfig:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = MTCNNDataSetConfig()
    train_config = MTCNNTrainConfig()
    eval_config = MTCNNEvalConfig()
    vis_config = MTCNNVisualizationConfig()
