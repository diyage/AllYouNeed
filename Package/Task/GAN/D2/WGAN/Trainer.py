from ..Dev import DevTrainer
from .Model import WGANModel
from .Loss import WGANLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


class WGANTrainer(DevTrainer):
    def __init__(
            self,
            model: WGANModel,
            noise_channel: int,
            clamp_value_tuple: tuple,
            train_g_frequency: int = 1,
            train_d_frequency: int = 1,

    ):
        super().__init__(
            train_g_frequency,
            train_d_frequency
        )
        self.model = model
        self.noise_channel = noise_channel
        self.clamp_value_tuple = clamp_value_tuple

    def clamp_weight_d(
            self,
    ):
        for p in self.model.discriminator.parameters():
            p.data.clamp_(
                self.clamp_value_tuple[0],
                self.clamp_value_tuple[1]
            )

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: WGANLoss,
            optimizer_g: torch.optim.Optimizer,
            optimizer_d: torch.optim.Optimizer,
            desc: str = '',
            now_epoch: int = 0,
            *args,
            **kwargs
    ) -> dict:

        d_device = next(self.model.discriminator.parameters()).device
        loss_dict_vec = {}
        for batch_id, real_images in enumerate(tqdm(data_loader_train,
                                                    desc=desc,
                                                    position=0)):
            self.model.generator.train()
            self.model.discriminator.train()

            real_images: torch.Tensor = real_images.to(d_device)

            if (batch_id + 1) % self.train_d_frequency == 0:
                fake_images = self.model.get_fake_images(
                    real_images.shape[0],
                    self.noise_channel
                )
                fake_predict = self.model.get_predict(fake_images)
                real_predict = self.model.get_predict(real_images)
                d_loss_res = loss_func.compute_discriminator_loss(
                    real_predict,
                    fake_predict
                )
                if not isinstance(d_loss_res, dict):
                    raise RuntimeError(
                        'You have not use our provided loss func, please overwrite method train_detector_one_epoch'
                    )
                else:
                    loss = d_loss_res['total_loss']
                    optimizer_d.zero_grad()
                    optimizer_g.zero_grad()
                    loss.backward()
                    optimizer_d.step()
                    self.clamp_weight_d()

                    for d_key, val in d_loss_res.items():
                        key = 'discriminator_' + d_key
                        if key not in loss_dict_vec.keys():
                            loss_dict_vec[key] = []
                        loss_dict_vec[key].append(val.item())

            if (batch_id + 1) % self.train_g_frequency == 0:
                fake_images = self.model.get_fake_images(
                    real_images.shape[0],
                    self.noise_channel
                )
                fake_predict = self.model.get_predict(fake_images)
                g_loss_res = loss_func.compute_generator_loss(
                    fake_predict
                )
                if not isinstance(g_loss_res, dict):
                    raise RuntimeError(
                        'You have not use our provided loss func, please overwrite method train_detector_one_epoch'
                    )
                else:
                    loss = g_loss_res['total_loss']
                    optimizer_d.zero_grad()
                    optimizer_g.zero_grad()
                    loss.backward()
                    optimizer_g.step()

                    for g_key, val in g_loss_res.items():
                        key = 'generator_' + g_key
                        if key not in loss_dict_vec.keys():
                            loss_dict_vec[key] = []
                        loss_dict_vec[key].append(val.item())

        loss_dict = {}
        for key, val in loss_dict_vec.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict

