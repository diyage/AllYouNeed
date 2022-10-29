from Package.DataSet.ForObjectDetection.VOC_R import *
from Package.Task.ObjectDetection.D2.YOLO.V3 import *
from Demo.yolo_v3.other import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 1

    config = YOLOV3Config()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    dark_net_53 = get_backbone_darknet_53(
        '/home/dell/PycharmProjects/YOLO/pre_trained/darknet53_75.42.pth'
    )
    # mAP is

    net = YOLOV3Model(dark_net_53)

    net.to(config.train_config.device)

    """
            get data
    """
    voc_train_loader = get_voc_data_loader(
        config.data_config.root_path,
        ['2007', '2012'],
        config.data_config.image_size,
        config.train_config.batch_size,
        train=True,
        num_workers=config.train_config.num_workers,
        mean=config.data_config.mean,
        std=config.data_config.std
    )
    voc_test_loader = get_voc_data_loader(
        config.data_config.root_path,
        ['2007'],
        config.data_config.image_size,
        config.train_config.batch_size,
        train=False,
        num_workers=config.train_config.num_workers,
        mean=config.data_config.mean,
        std=config.data_config.std
    )

    helper = YOLOV3Helper(
        net,
        config
    )
    helper.go(voc_train_loader, voc_test_loader)

