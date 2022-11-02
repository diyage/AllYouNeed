from Demo.wgan.other import *
from Package.Task.GAN.D2.WGAN import *
from Package.DataSet.ForGAN.Cartoon import get_cartoon_loader
import albumentations as alb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    """
    W-GAN demo, changed :
        1) loss, please see WGANLoss
        2) structure, do not use sigmoid behind(discriminator)
        3) optimizer, use RMS prop
        4) clip weight parameter(s) of discriminator
    """
    GPU_ID = 0

    config = WGANConfig()

    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    model = WGANModel(
        d_net=DiscriminatorNet(),
        g_net=GeneratorNet(config.train_config.noise_channel)
    )

    model.to(config.train_config.device)

    """
            get data
    """
    trans_train = alb.Compose([
        alb.HueSaturationValue(),
        alb.HorizontalFlip(),
        alb.Resize(*config.data_config.image_size),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ])

    train_loader = get_cartoon_loader(
        config.data_config.root,
        trans=trans_train,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers
    )

    helper = WGANHelper(
        model,
        config,
        restore_epoch=-1
    )

    helper.go(train_loader)
