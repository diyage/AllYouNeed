class OriginalGANDataSetConfig:
    root: str = '/home/dell/data/DataSet/Cartoon'
    mean: list = [0.5, 0.5, 0.5]
    std: list = [0.5, 0.5, 0.5]
    image_size: tuple = (96, 96)


class OriginalGANTrainConfig:
    d_lr: float = 1e-4
    g_lr: float = 1e-4
    train_g_frequency: int = 5
    train_d_frequency: int = 1
    max_epoch_for_train: int = 1000
    num_workers: int = 8
    device: str = 'cuda:0'
    batch_size = 128
    noise_channel: int = 100


class OriginalGANEvalConfig:
    eval_frequency = 10


class OriginalGANVisualizationConfig:
    generate_fake_image_num: int = 128


class OriginalGANConfig:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = OriginalGANDataSetConfig()
    train_config = OriginalGANTrainConfig()
    eval_config = OriginalGANEvalConfig()
    vis_config = OriginalGANVisualizationConfig()
