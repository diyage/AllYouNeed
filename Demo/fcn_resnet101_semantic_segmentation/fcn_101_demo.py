from Package.DataSet.ForSegmentation.VOC import get_voc_for_all_tasks_loader
from Demo.fcn_resnet101_semantic_segmentation.other import *
import albumentations as alb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 0
    """
        set config
    """
    config = FCNResNet101Config()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)
    config.train_config.lr = 1e-3
    """
        build model
    """
    model = FCNResNet101Model(
        get_fcn_resnet101(pre_trained=False, num_classes=21)
    )
    model.to(config.train_config.device)

    """
        get data
    """
    trans_train = alb.Compose([
        alb.HueSaturationValue(),
        alb.Rotate(),
        alb.RandomBrightnessContrast(),
        alb.ColorJitter(),
        alb.Blur(),
        alb.GaussNoise(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    trans_test = alb.Compose([
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    voc_train_loader = get_voc_for_all_tasks_loader(
        config.data_config.root_path,
        ['2007', '2012'],
        train=True,
        image_size=config.data_config.image_size[0],
        kind_name_to_color=config.data_config.KIND_NAME_TO_COLOR,
        mean=config.data_config.mean,
        std=config.data_config.std,
        trans_form=trans_train,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
        use_bbox=False,
        use_mask_type=-1
    )
    voc_test_loader = get_voc_for_all_tasks_loader(
        config.data_config.root_path,
        ['2007'],
        train=False,
        image_size=config.data_config.image_size[0],
        kind_name_to_color=config.data_config.KIND_NAME_TO_COLOR,
        mean=config.data_config.mean,
        std=config.data_config.std,
        trans_form=trans_test,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
        use_bbox=False,
        use_mask_type=-1
    )

    helper = FCNResNet101Helper(
        model,
        config,
        restore_epoch=-1
    )
    helper.go(voc_train_loader, voc_test_loader)

