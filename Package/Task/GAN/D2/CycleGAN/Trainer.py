from ..Dev import DevTrainer
from .Model import CycleGANModel
from .Loss import CycleGANLoss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class CycleGANTrainer(DevTrainer):
    def __init__(
            self,
            model: CycleGANModel,
            train_g_frequency: int = 1,
            train_d_frequency: int = 1,
    ):
        super().__init__(
            train_g_frequency,
            train_d_frequency
        )
        self.model = model

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: CycleGANLoss,
            optimizer_g: torch.optim.Optimizer,
            optimizer_d_a: torch.optim.Optimizer,
            optimizer_d_b: torch.optim.Optimizer,
            desc: str = '',
            now_epoch: int = 0,
            *args,
            **kwargs
    ) -> dict:
        device = next(self.model.g_a_to_b.parameters()).device

        loss_dict_vec = {}
        for batch_id, (a, b) in enumerate(tqdm(data_loader_train,
                                               desc=desc,
                                               position=0)):
            a: torch.Tensor = a.to(device)
            b: torch.Tensor = b.to(device)
            self.model.train()

            if (batch_id + 1) % self.train_d_frequency == 0:
                res = self.model.get_compute_discriminator_loss_need_info(
                    a,
                    b
                )
                d_loss_res = loss_func.compute_discriminator_loss(
                    res
                )
                if not isinstance(d_loss_res, dict):
                    raise RuntimeError(
                        'You have not use our provided loss func, please overwrite method train_one_epoch'
                    )
                else:
                    loss_d_a: torch.Tensor = d_loss_res['loss_d_a']
                    self.model.zero_grad()
                    loss_d_a.backward()
                    optimizer_d_a.step()

                    loss_d_b: torch.Tensor = d_loss_res['loss_d_b']
                    self.model.zero_grad()
                    loss_d_b.backward()
                    optimizer_d_b.step()

                    for d_key, val in d_loss_res.items():
                        key = 'discriminator_' + d_key
                        if key not in loss_dict_vec.keys():
                            loss_dict_vec[key] = []
                        loss_dict_vec[key].append(val.item())

            if (batch_id + 1) % self.train_g_frequency == 0:
                res = self.model.get_compute_generator_loss_need_info(
                    a,
                    b
                )
                g_loss_res = loss_func.compute_generator_loss(
                    res
                )
                if not isinstance(g_loss_res, dict):
                    raise RuntimeError(
                        'You have not use our provided loss func, please overwrite method train_one_epoch'
                    )
                else:
                    loss = g_loss_res['total_loss']
                    self.model.zero_grad()
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
