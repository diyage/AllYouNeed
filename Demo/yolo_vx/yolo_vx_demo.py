"""
之前实现的 YOLO 系列代码（截止至 YOLO v5）在宏观上保证了和原文的一致，但是一些个中细节有些许差别
此外，由于绝大部分的代码都是由自己撰写的，因此效率会有些折扣，包括一些本人觉得suboptimal的地方后
续都会改正

自 YOLO X之后，所有代码都尽量找到开源代码，并且自己实现一份，进行对比，保持一致
解决一些之前调试的时候经常被诟病的问题
"""

from Package.DataSet.ForObjectDetection.COCO import get_coco_data_loader
from Package.Task.ObjectDetection.D2.YOLO.VX import *
from Demo.yolo_vx.other import *
from PIL import ImageFile
import albumentations as alb
import torch.nn as nn

ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 0

    config = YOLOVXConfig()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)
    config.train_config.num_workers = 8
    config.train_config.batch_size = 8
    net = YOLOVXModel(
        YOLOVXPanFpn(),
        cls_num=len(config.data_config.kinds_name),
        image_shrink_rate=config.data_config.image_shrink_rate
    )

    net.to(config.train_config.device)
    net = nn.DataParallel(
        net,
        device_ids=[0, 1]
    )
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
        alb.Resize(config.data_config.image_size, config.data_config.image_size),
        alb.Normalize(
            config.data_config.mean,
            config.data_config.std
        )

    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    trans_test = alb.Compose([
        alb.Resize(config.data_config.image_size, config.data_config.image_size),
        alb.Normalize(
            config.data_config.mean,
            config.data_config.std
        )

    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    coco_train_loader = get_coco_data_loader(
        config.data_config.root_path,
        config.data_config.years,
        train=True,
        image_size=config.data_config.image_size,
        trans_form=trans_train,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
        use_mosaic=config.train_config.use_mosaic,
        use_label_type=config.data_config.use_label_type
    )
    coco_test_loader = get_coco_data_loader(
        config.data_config.root_path,
        config.data_config.years,
        train=False,
        image_size=config.data_config.image_size,
        trans_form=trans_test,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
        use_mosaic=False,
        use_label_type=config.data_config.use_label_type
    )
    config.init_lr_mapping(
        config.train_config.max_epoch_on_detector,
        len(coco_train_loader),
        config.train_config.reach_base_lr_cost_epoch,
        config.train_config.lr,
    )
    helper = YOLOVXHelper(
        net,
        config,
        restore_epoch=-1
    )

    helper.go(coco_train_loader, coco_test_loader)


