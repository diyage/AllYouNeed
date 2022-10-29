class ArcFaceConfig:
    lr: float = 1e-3
    ABS_PATH: str = '/home/dell/data2/models/'
    use_l2_norm_feature: bool = True
    compute_threshold_num: int = 1000
    distance_type: str = 'cosine_similarity'
    max_epoch_for_train: int = 200
    warm_up_end_epoch: int = 5
    eval_frequency: int = 10
    device: str = 'cpu'
    feature_num: int = 512
    lfw_class_num: int = 5749
    margin: float = 0.5
    scale: float = 64.0
    root = '/home/dell/data/DataSet/LFW112x112/images'
    pair_path = '/home/dell/data/DataSet/LFW112x112/pairs.txt'
    batch_size: int = 128
    num_workers: int = 8
