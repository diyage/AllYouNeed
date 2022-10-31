
class FCNResNet101DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    # data set root dir
    image_size: tuple = (608, 608)
    KIND_NAME_TO_COLOR = {
        'background': (0, 0, 0),
        'aeroplane': (0, 0, 128),
        'bicycle': (0, 128, 0),
        'bird': (0, 128, 128),
        'boat': (128, 0, 0),
        'bottle': (128, 0, 128),
        'bus': (128, 128, 0),
        'car': (128, 128, 128),
        'cat': (0, 0, 64),
        'chair': (0, 0, 192),
        'cow': (0, 128, 64),
        'dining table': (0, 128, 192),
        'dog': (128, 0, 64),
        'horse': (128, 0, 192),
        'motorbike': (128, 128, 64),
        'person': (128, 128, 192),
        'potted plant': (0, 64, 0),
        'sheep': (0, 64, 128),
        'sofa': (0, 192, 0),
        'train': (0, 192, 128),
        'tv monitor': (128, 64, 0),
        # 'bordering region': (192, 224, 224),
    }
    mean = [0.406, 0.456, 0.485]

    std = [0.225, 0.224, 0.229]


class FCNResNet101TrainConfig:
    max_epoch_on_detector = 200
    num_workers: int = 4
    device: str = 'cuda:0'
    batch_size = 4
    lr: float = 1e-3
    warm_up_end_epoch: int = 5


class FCNResNet101EvalConfig:
    eval_frequency = 10


class FCNResNet101Config:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = FCNResNet101DataSetConfig()
    train_config = FCNResNet101TrainConfig()
    eval_config = FCNResNet101EvalConfig()


