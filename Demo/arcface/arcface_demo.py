from Package.DataSet.ForFacialRecognition.LFW import get_lfw_loader, get_data_pair
from Package.Task.FacialRecognition.D2.ArcFace import *
from Demo.arcface.other import *
import albumentations as alb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 1

    config = ArcFaceConfig()

    config.device = 'cuda:{}'.format(GPU_ID)

    model = ArcFaceModel(
        MobileFaceNet(config.feature_num),
        feature_num=config.feature_num,
        class_num=config.lfw_class_num,
        margin=config.margin,
        scale=config.scale
    )

    model.to(config.device)

    """
            get data
    """
    trans_train = alb.Compose([
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

    lfw_train_loader = get_lfw_loader(
        config.root,
        train=True,
        trans=trans_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    lfw_test_loader = get_lfw_loader(
        config.root,
        train=False,
        trans=trans_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    data_pair = get_data_pair(
        config.pair_path,
        config.root
    )

    helper = ArcFaceHelper(
        model,
        config
    )

    helper.go(lfw_train_loader, lfw_test_loader, data_pair)


