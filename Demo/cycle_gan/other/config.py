class CycleGANDataSetConfig:
    root: str = '/home/dell/data/DataSet/Grumpifycat'
    mean: list = [0.5, 0.5, 0.5]
    std: list = [0.5, 0.5, 0.5]
    image_size_tuple: tuple = (256, 256)
    strict_pair: bool = False


class CycleGANTrainConfig:
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999

    n_residual_blocks: int = 9
    input_nc_A: int = 3
    input_nc_B: int = 3
    n_D_layers: int = 4

    lambda_cycle: float = 10
    lambda_identity: float = 0.5

    train_g_frequency: int = 5
    train_d_frequency: int = 1

    device: str = 'cuda:0'
    batch_size = 1
    max_epoch_for_train: int = 10000
    num_workers: int = 8


class CycleGANEvalConfig:
    eval_frequency = 10


class CycleGANVisualizationConfig:
    pass


class CycleGANConfig:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = CycleGANDataSetConfig()
    train_config = CycleGANTrainConfig()
    eval_config = CycleGANEvalConfig()
    vis_config = CycleGANVisualizationConfig()
