from ..Dev import DevVisualizer
from .Model import CycleGANModel
from .Tools import CycleGANTool
from typing import List
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from Package.BaseDev import CV2
import numpy as np


class CycleGANVisualizer(DevVisualizer):
    def __init__(
            self,
            model: CycleGANModel,
            mean: List[float],
            std: List[float],
    ):
        super().__init__(
            model,
            mean,
            std
        )
        self.model: CycleGANModel = model
        """
        just want to call method through(dot)
        """

    def show_generate_results(
            self,
            data_loader: DataLoader,
            saved_dir: str,
            desc: str = 'show generate result',
            *args,
            **kwargs
    ):
        os.makedirs(saved_dir, exist_ok=True)
        device = next(self.model.g_a_to_b.parameters()).device
        for batch_id, (real_a, real_b) in enumerate(
            tqdm(
                data_loader,
                desc=desc,
                position=0
            )
        ):
            real_a: torch.Tensor = real_a.to(device)
            real_b: torch.Tensor = real_b.to(device)
            self.model.eval()
            fake_a: torch.Tensor = self.model.g_b_to_a(real_b)
            fake_b: torch.Tensor = self.model.g_a_to_b(real_a)
            for ind in range(real_a.shape[0]):
                a: np.ndarray = CycleGANTool.image_tensor_to_np(
                    real_a[ind],
                    self.mean,
                    self.std
                )
                CV2.imwrite(
                    '{}/{}_{}_style_a.png'.format(
                        saved_dir,
                        batch_id,
                        ind,
                    ),
                    a
                )

                a_to_b: np.ndarray = CycleGANTool.image_tensor_to_np(
                    fake_b[ind],
                    self.mean,
                    self.std
                )
                CV2.imwrite(
                    '{}/{}_{}_style_a_to_b.png'.format(
                        saved_dir,
                        batch_id,
                        ind,
                    ),
                    a_to_b
                )

                b: np.ndarray = CycleGANTool.image_tensor_to_np(
                    real_b[ind],
                    self.mean,
                    self.std
                )
                CV2.imwrite(
                    '{}/{}_{}_style_b.png'.format(
                        saved_dir,
                        batch_id,
                        ind,
                    ),
                    b
                )

                b_to_a: np.ndarray = CycleGANTool.image_tensor_to_np(
                    fake_a[ind],
                    self.mean,
                    self.std
                )
                CV2.imwrite(
                    '{}/{}_{}_style_b_to_a.png'.format(
                        saved_dir,
                        batch_id,
                        ind,
                    ),
                    b_to_a
                )
