from ..Dev import DevTrainer
from .Model import MTCNNModel
from .Loss import MTCNNLoss
from .Tools import MTCNNTool
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class MTCNNTrainer(DevTrainer):
    def __init__(
            self,
            model: MTCNNModel,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.device

    def make_target(
            self,
            image_type: int,
            key_point: torch.Tensor,
            positive_position_offset: torch.Tensor,
            part_position_offset: torch.Tensor
    ) -> dict:
        target: dict = MTCNNTool.make_target(
            image_type,
            key_point,
            positive_position_offset,
            part_position_offset
        )
        for key, val in target.items():
            if isinstance(val, torch.Tensor):
                target[key] = val.to(self.device)
        return target

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: MTCNNLoss,
            optimizer: torch.optim.Optimizer,
            now_epoch: int,
            max_epoch: int,
            train_net_type: str,
            *args,
            **kwargs
    ) -> dict:
        assert train_net_type in ['p', 'r', 'o']
        self.model.train()
        now_epoch_loss_dict = {}
        for batch_id, res in enumerate(tqdm(data_loader_train,
                                            desc='Training {}-Net with epoch {}/{}'.format(
                                                train_net_type,
                                                now_epoch,
                                                max_epoch
                                            ),
                                            position=0)):

            img, key_point, positive, part = res
            img: torch.Tensor = img.to(self.device)  # (batch, 3, h, w, 6)

            key_point: torch.Tensor = key_point.to(self.device)  # (batch, 10)
            positive_position_offset: torch.Tensor = positive.to(self.device)  # (batch, 4)
            part_position_offset: torch.Tensor = part.to(self.device)  # (batch, 4)

            now_batch_loss_dict = {
                'total_loss': 0,
                'key_point_loss': 0,
                'cls_loss': 0,
                'position_loss': 0,
            }

            for image_type in range(6):
                now_type_image = img[..., image_type]  # (batch, 3, h, w)
                predict = self.model(now_type_image, use_net_type=train_net_type)
                target = self.make_target(
                    image_type,
                    key_point,
                    positive_position_offset,
                    part_position_offset
                )
                temp_loss_dict: dict = loss_func(predict, target, train_net_type=train_net_type, image_type=image_type)

                for loss_type, loss_val in temp_loss_dict.items():
                    now_batch_loss_dict[loss_type] += loss_val

            loss: torch.Tensor = now_batch_loss_dict['total_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key, val in now_batch_loss_dict.items():
                if key not in now_epoch_loss_dict.keys():
                    now_epoch_loss_dict[key] = []
                now_epoch_loss_dict[key].append(val.item())

        loss_dict = {}
        for key, val in now_epoch_loss_dict.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict
