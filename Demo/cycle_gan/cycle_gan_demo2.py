from Demo.cycle_gan.other import *
from Package.Task.GAN.D2.CycleGAN import *
from Package.DataSet.ForGAN.Portrait import get_portrait_data_loader
import albumentations as alb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':

    GPU_ID = 1

    config = CycleGANConfig()
    config.data_config.root = '/home/dell/data/DataSet/Portrait'
    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    g_a_to_b, d_b, g_b_to_a, d_a = create_nets(
        config.train_config.input_nc_A,
        config.train_config.input_nc_B,
        config.train_config.n_D_layers,
        config.train_config.n_residual_blocks
    )
    model = CycleGANModel(
        g_b_to_a=g_b_to_a,
        g_a_to_b=g_a_to_b,
        d_a=d_a,
        d_b=d_b
    )

    model.to(config.train_config.device)

    """
            get data
    """
    trans_train = alb.Compose([
        alb.HueSaturationValue(),
        alb.HorizontalFlip(),
        alb.Resize(*config.data_config.image_size_tuple),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ])

    train_loader = get_portrait_data_loader(
        config.data_config.root,
        train=True,
        transform=trans_train,
        strict_pair=config.data_config.strict_pair,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers
    )
    trans_test = alb.Compose([
        alb.Resize(*config.data_config.image_size_tuple),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ])

    test_loader = get_portrait_data_loader(
        config.data_config.root,
        train=False,
        transform=trans_test,
        strict_pair=config.data_config.strict_pair,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers
    )

    helper = CycleGANHelper(
        model,
        config,
        restore_epoch=-1
    )

    helper.go(train_loader, test_loader)
