from Package.Task.FacialRecognition.D2.Dev import DevTrainer
from .Model import ArcFaceModel
from .Tools import ArcFaceTool
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from Package.Optimizer.WarmUp import WarmUpCosineAnnealOptimizer


class ArcFaceTrainer(DevTrainer):
    def __init__(
            self,
            model: ArcFaceModel,
    ):
        super().__init__(
            model
        )

    def make_targets(
            self,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        return ArcFaceTool.make_target(targets).to(self.device)

    def train_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: nn.Module,
            optimizer: WarmUpCosineAnnealOptimizer,
            desc: str = '',
            now_epoch: int = 0,
    ):
        loss_dict_vec = {}
        max_batch_ind = len(data_loader_train)

        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            optimizer.warm(
                now_epoch,
                batch_id,
                max_batch_ind
            )
            self.model.train()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            output = self.model(images, target=targets)

            """
            be careful, output is a dict.
            """
            loss_res = loss_func(output, targets)
            if not isinstance(loss_res, dict):
                print('You have not use our provided loss func...')
                pass
            else:
                loss = loss_res['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for key, val in loss_res.items():
                    if key not in loss_dict_vec.keys():
                        loss_dict_vec[key] = []
                    loss_dict_vec[key].append(val.item())

        loss_dict = {}
        for key, val in loss_dict_vec.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict
