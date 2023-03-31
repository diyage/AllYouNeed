from Package.DataSet.ForObjectDetection.COCO import get_coco_data_loader
from Package.Task.ObjectDetection.D2.YOLO.V5 import *
from Demo.yolo_v5.other import *
from PIL import ImageFile
import albumentations as alb


ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 0

    config = YOLOV5Config()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    net = YOLOV5Model(3, 32, len(config.data_config.kinds_name), single_anchor_num=config.data_config.single_an)

    net.to(config.train_config.device)

    """
            get data
    """
    trans_train = alb.Compose([
        alb.GaussNoise(),
        alb.HueSaturationValue(),
        alb.RandomBrightnessContrast(),
        # alb.ColorJitter(),
        # alb.Blur(3),
        alb.HorizontalFlip(),
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            config.data_config.mean,
            config.data_config.std
        )

    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    trans_test = alb.Compose([
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            config.data_config.mean,
            config.data_config.std
        )

    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    coco_train_loader = get_coco_data_loader(
        config.data_config.root_path,
        ['2014', '2017'],
        train=True,
        image_size=config.data_config.image_size[0],
        trans_form=trans_train,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,

    )
    coco_test_loader = get_coco_data_loader(
        config.data_config.root_path,
        ['2014', '2017'],
        train=False,
        image_size=config.data_config.image_size[0],
        trans_form=trans_test,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,

    )

    helper = YOLOV5Helper(
        net,
        config,
        restore_epoch=10
    )

    helper.go(coco_train_loader, coco_test_loader)

