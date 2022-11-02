from .config import CycleGANConfig
from Package.Task.GAN.D2.CycleGAN import *
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader


class CycleGANHelper:
    def __init__(
            self,
            model: CycleGANModel,
            config: CycleGANConfig,
            restore_epoch: int = -1
    ):
        self.model = model  # type: CycleGANModel
        self.device = next(model.parameters()).device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = CycleGANTrainer(
            model,
            self.config.train_config.train_g_frequency,
            self.config.train_config.train_d_frequency,
        )

        self.visualizer = CycleGANVisualizer(
            model,
            self.config.data_config.mean,
            self.config.data_config.std,
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
            data_loader_test: DataLoader,
            epoch: int
    ):
        with torch.no_grad():
            saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.visualizer.show_generate_results(
                data_loader_test,
                saved_dir,
                desc='[show generate results]',
            )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = CycleGANLoss(
            self.config.data_config.image_size_tuple,
            self.config.train_config.n_D_layers,
            self.config.train_config.lambda_cycle,
            self.config.train_config.lambda_identity
        )

        optimizer_g = torch.optim.Adam([*self.model.g_a_to_b.parameters(),
                                        *self.model.g_b_to_a.parameters()],
                                       lr=self.config.train_config.lr,
                                       betas=(self.config.train_config.b1, self.config.train_config.b2), )

        optimizer_d_a = torch.optim.Adam(
            self.model.d_a.parameters(),
            lr=self.config.train_config.lr,
            betas=(self.config.train_config.b1, self.config.train_config.b2)
        )
        optimizer_d_b = torch.optim.Adam(
            self.model.d_b.parameters(),
            lr=self.config.train_config.lr,
            betas=(self.config.train_config.b1, self.config.train_config.b2)
        )

        for epoch in tqdm(range(self.restore_epoch + 1, self.config.train_config.max_epoch_for_train),
                          desc='training Cycle-GAN',
                          position=0):

            loss_dict = self.trainer.train_one_epoch(
                data_loader_train,
                loss_func,
                optimizer_g,
                optimizer_d_a,
                optimizer_d_b,
                desc='[train for Cycle-GAN epoch: {}/{}]'.format(
                    epoch,
                    self.config.train_config.max_epoch_for_train - 1
                ),
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
                self.show_generate_results(data_loader_test, epoch)
