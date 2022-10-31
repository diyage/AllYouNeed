from abc import abstractmethod
from typing import Union
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from .predictor import DevPredictor
from .model import DevModel
from .tools import DevTool
import matplotlib.pyplot as plt
from Package.BaseDev import BaseVisualizer, CV2
from typing import List


class DevVisualizer(BaseVisualizer):
    def __init__(
            self,
            model: DevModel,
            predictor: DevPredictor,
            image_mean: List[float],
            image_std: List[float],
            kind_name_to_color: dict
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.predictor = predictor
        self.image_mean = image_mean
        self.image_std = image_std
        self.KIND_NAME_TO_COLOR = kind_name_to_color

    def make_targets(
            self,
            labels: List[np.ndarray],
    ):
        targets = DevTool.make_target(
            labels,
        )

        return targets.to(self.device)

    def mix_mask(
            self,
            mask: np.ndarray
    ):
        """

        :param mask: h * w *  mask_num
        :return:
        """
        h, w, c = mask.shape[0], mask.shape[1], 3

        mix_mask = np.zeros(shape=(h, w, c), dtype=np.float32)

        for i, kind_name in enumerate(self.KIND_NAME_TO_COLOR.keys()):
            kind_color = self.KIND_NAME_TO_COLOR.get(kind_name)  # (3, )

            kind_mask = mask[:, :, i]  # (H, W)
            kind_mask = np.expand_dims(kind_mask, axis=-1).repeat(c, axis=-1)  # (H, W, c)

            kind_mask = kind_mask * np.array(kind_color, dtype=np.float32)
            mix_mask += kind_mask

        return mix_mask.astype(np.uint8)

    def show_(
            self,
            image: np.ndarray,
            pre_mask_vec: np.ndarray,
            gt_mask_vec: np.ndarray,
            saved_file_name: str

    ):
        image = image.copy().astype(np.float32)
        pre_mask_vec = pre_mask_vec.copy().astype(np.float32)
        gt_mask_vec = gt_mask_vec.copy().astype(np.float32)

        image = CV2.cvtColorToRGB(image.astype(np.uint8))
        pre_mask_vec = self.mix_mask(pre_mask_vec)
        gt_mask_vec = self.mix_mask(gt_mask_vec)

        ax1 = plt.subplot(1, 3, 1)  # type: plt.Axes
        plt.imshow(image)
        ax1.set_title('image')

        ax2 = plt.subplot(1, 3, 2)  # type: plt.Axes
        plt.imshow(gt_mask_vec)
        ax2.set_title('target')

        ax3 = plt.subplot(1, 3, 3)  # type: plt.Axes
        plt.imshow(pre_mask_vec)
        ax3.set_title('predict')

        # ax.legend()
        plt.savefig(saved_file_name, bbox_inches='tight', pad_inches=0.0)
        plt.close()

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):
        os.makedirs(saved_dir, exist_ok=True)
        for batch_ind, (images, objects, masks) in enumerate(tqdm(data_loader_test,
                                                                  desc=desc,
                                                                  position=0)):
            if batch_ind == 10:
                break

            self.model.eval()
            images = images.to(self.device)

            targets = self.make_targets(masks)

            output = self.model(images)

            gt_decode = self.predictor.decode_target(targets)  # type: np.ndarray
            pre_decode = self.predictor.decode_predict(output)  # type: np.ndarray

            for image_ind in range(images.shape[0]):
                image_i = images[image_ind].permute(1, 2, 0).cpu().detach().numpy()
                image_i = image_i * np.array(self.image_std) + np.array(self.image_mean)
                image_i = image_i * 255.0
                image_i = image_i.astype(np.uint8)

                pre_decode_mask = pre_decode[image_ind]
                gt_decode_mask = gt_decode[image_ind]

                self.show_(
                    image_i,
                    pre_decode_mask,
                    gt_decode_mask,
                    saved_file_name='{}/{}_{}_segmentation.png'.format(saved_dir, batch_ind, image_ind)
                )
