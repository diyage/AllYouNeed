from Package.DataSet.ForFacialRecognition.MS1M import get_ms1m_data_loader
from Package.DataSet.ForFacialRecognition.LFW import get_lfw_loader, get_data_pair
from Package.Task.FacialRecognition.D2.ArcFace import *
from Demo.arcface.other import *
import albumentations as alb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 1

    config = ArcFaceConfig()

    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    model = ArcFaceModel(
        MobileFaceNet(config.train_config.feature_num),
        feature_num=config.train_config.feature_num,
        class_num=config.data_set_config.ms1m_class_num,
        margin=config.train_config.margin,
        scale=config.train_config.scale
    )

    model.to(config.train_config.device)

    """
            get data
    """
    trans_train = alb.Compose([
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        alb.Rotate(limit=(-15, 15)),
        alb.HorizontalFlip(),
        alb.Resize(112, 112),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    trans_test = alb.Compose([
        alb.Resize(112, 112),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    ms1m_train_loader = get_ms1m_data_loader(
        config.data_set_config.ms1m_root,
        trans=trans_train,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers
    )

    lfw_test_loader = get_lfw_loader(
        config.data_set_config.lfw_root,
        train=False,
        trans=trans_test,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers
    )

    data_pair = get_data_pair(
        config.data_set_config.lfw_pair_path,
        config.data_set_config.lfw_root
    )

    helper = ArcFaceHelper(
        model,
        config
    )
    helper.restore(60)
    helper.go(ms1m_train_loader, lfw_test_loader, data_pair)


