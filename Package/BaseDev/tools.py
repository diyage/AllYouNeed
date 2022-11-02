"""
This packet is the most-most-most core development tool.
It will serve for all other development tools.
You could use it to define everything !!!
"""
import torch
import numpy as np
from .cv2_ import *
from typing import Union
from abc import abstractmethod


class BaseTool:

    @staticmethod
    @abstractmethod
    def make_target(
            *args,
            **kwargs
    ):
        """
        create target used for computing loss.
        You may have one question: where is make_predict ?
        Method --make_predict-- is just __call__(or forward, little incorrect) of nn.Module !!!
        So predict is just the output of model(nn.Module you define).
        Please see model.BaseModel
        """
        pass

    @staticmethod
    @abstractmethod
    def split_target(
            *args,
            **kwargs
    ):
        """
        sometimes ,you need split target(e.g., object detection)
        """
        pass

    @staticmethod
    @abstractmethod
    def split_predict(
            *args,
            **kwargs
    ):
        """
        sometimes ,you need split your predict(e.g., object detection)
        """
        pass

    @staticmethod
    def image_np_to_tensor(
            image: np.ndarray,
            mean=[0.406, 0.456, 0.485],
            std=[0.225, 0.224, 0.229],
    ) -> torch.Tensor:
        # image is a BGR uint8 image
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        # image = CV2.cvtColorToRGB(image)  # (H, W, C)
        image = ((1.0 * image / 255.0) - mean) / std  # (H, W, C)
        image = np.transpose(image, axes=(2, 0, 1))  # (C, H, W)
        return torch.tensor(image, dtype=torch.float32)

    @staticmethod
    def image_tensor_to_np(
            img: torch.Tensor,
            mean=[0.406, 0.456, 0.485],
            std=[0.225, 0.224, 0.229]
    ) -> np.ndarray:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        img = img.cpu().detach().numpy().copy()  # type:np.ndarray
        # (C, H, W)
        img = np.transpose(img, axes=(1, 2, 0))  # type:np.ndarray
        # (H, W, C)
        img = ((img * std) + mean) * 255.0
        img = img.astype(np.uint8).copy()  # type:np.ndarray

        return img

