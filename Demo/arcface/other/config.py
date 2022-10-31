class ArcFaceDataSetConfig:
    lfw_class_num: int = 5749
    ms1m_class_num: int = 85164
    lfw_root: str = '/home/dell/data/DataSet/LFW112x112/images'
    lfw_pair_path = '/home/dell/data/DataSet/LFW112x112/pairs.txt'
    ms1m_root: str = '/home/dell/data/DataSet/faces_ms1m_112x112/images'


class ArcFaceTrainConfig:
    lr: float = 1e-3
    warm_up_end_epoch: int = 5
    max_epoch_for_train: int = 200
    device: str = 'cpu'

    feature_num: int = 512
    margin: float = 0.5
    scale: float = 64.0
    batch_size: int = 256
    num_workers: int = 8


class ArcFaceEvalConfig:
    use_l2_norm_feature: bool = True
    compute_threshold_num: int = 1000
    distance_type: str = 'cosine_similarity'
    eval_frequency: int = 10


class ArcFaceConfig:
    data_set_config = ArcFaceDataSetConfig()
    train_config = ArcFaceTrainConfig()
    eval_config = ArcFaceEvalConfig()
    ABS_PATH: str = '/home/dell/data2/models/'
