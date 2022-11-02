class WGANDataSetConfig:
    root: str = '/home/dell/data/DataSet/Cartoon'
    mean: list = [0.5, 0.5, 0.5]
    std: list = [0.5, 0.5, 0.5]
    image_size: tuple = (96, 96)


class WGANTrainConfig:
    d_lr: float = 1e-4
    g_lr: float = 1e-4
    train_g_frequency: int = 5
    train_d_frequency: int = 1
    max_epoch_for_train: int = 10000
    num_workers: int = 8
    device: str = 'cuda:0'
    batch_size = 128
    noise_channel: int = 100
    clip_value_tuple: tuple = (-0.01, +0.01)


class WGANEvalConfig:
    eval_frequency = 10


class WGANVisualizationConfig:
    generate_fake_image_num: int = 128


class WGANConfig:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = WGANDataSetConfig()
    train_config = WGANTrainConfig()
    eval_config = WGANEvalConfig()
    vis_config = WGANVisualizationConfig()
