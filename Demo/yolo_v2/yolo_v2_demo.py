from Package.DataSet.ForObjectDetection.VOC_R import *
from Package.Task.ObjectDetection.D2.YOLO.V2 import *
from Demo.yolo_v2.other import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    GPU_ID = 0

    config = YOLOV2Config()

    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    dark_net_19 = get_backbone_dark_net_19(
        '/home/dell/PycharmProjects/YOLO/pre_trained/darknet19_72.96.pth'
    )
    net = YOLOV2Model(dark_net_19)

    net.to(config.train_config.device)

    """
            get data
    """

    voc_train_loader = get_voc_data_loader(
        config.data_config.root_path,
        ['2007', '2012'],
        train=True,
        image_size=config.data_config.image_size,
        mean=config.data_config.mean,
        std=config.data_config.std,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
    )
    voc_test_loader = get_voc_data_loader(
        config.data_config.root_path,
        ['2007'],
        train=False,
        image_size=config.data_config.image_size,
        mean=config.data_config.mean,
        std=config.data_config.std,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
    )

    helper = YOLOV2Helper(
        net,
        config
    )
    helper.restore(190)
    helper.eval_map(voc_test_loader)
    # helper.go(voc_train_loader, voc_test_loader)

