from .config import WGANConfig
from Package.Task.GAN.D2.WGAN import *
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader


class WGANHelper:
    def __init__(
            self,
            model: WGANModel,
            config: WGANConfig,
            restore_epoch: int = -1
    ):
        self.model = model  # type: WGANModel
        self.device = next(model.parameters()).device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = WGANTrainer(
            model,
            self.config.train_config.noise_channel,
            self.config.train_config.clip_value_tuple,
            self.config.train_config.train_g_frequency,
            self.config.train_config.train_d_frequency
        )

        self.visualizer = WGANVisualizer(
            model,
            self.config.data_config.mean,
            self.config.data_config.std,
            self.config.train_config.noise_channel
        )

    def restore(
            self,
            epoch: int
    ):
        self.restore_epoch = epoch
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth/'
        saved_file_name = '{}/{}.pth'.format(saved_dir, epoch)
        self.model.load_state_dict(
            torch.load(saved_file_name)
        )

    def save(
            self,
            epoch: int
    ):
        # save model
        self.model.eval()
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.model.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

    def show_generate_results(
            self,
            epoch: int
    ):
        with torch.no_grad():
            saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.visualizer.show_generate_results(
                saved_dir,
                desc='[show generate results]',
                generate_num=self.config.vis_config.generate_fake_image_num
            )

    def go(
            self,
            data_loader_train: DataLoader,
    ):
        loss_func = WGANLoss()
        """
        RMS prop
        """
        optimizer_d = torch.optim.RMSprop(
            self.model.discriminator.parameters(),
            lr=self.config.train_config.d_lr,
        )
        optimizer_g = torch.optim.RMSprop(
            self.model.generator.parameters(),
            lr=self.config.train_config.g_lr,
        )

        for epoch in tqdm(range(self.restore_epoch + 1, self.config.train_config.max_epoch_for_train),
                          desc='training W-GAN',
                          position=0):

            loss_dict = self.trainer.train_one_epoch(
                data_loader_train,
                loss_func,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                desc='[train for w-GAN epoch: {}/{}]'.format(epoch, self.config.train_config.max_epoch_for_train - 1),
                now_epoch=epoch
            )

            print_info = '\n\nepoch: {} , loss info-->\n'.format(
                epoch
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if epoch % self.config.eval_config.eval_frequency == 0:
                # save model
                self.save(epoch)

                # show predict
                self.show_generate_results(epoch)
