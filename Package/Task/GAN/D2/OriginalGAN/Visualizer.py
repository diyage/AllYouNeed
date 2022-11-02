from Package.BaseDev import CV2
from ..Dev import DevVisualizer
from .Model import OriginalGANModel
from .Tools import OriginalGANTool
from typing import List
import torch
import os


class OriginalGANVisualizer(DevVisualizer):
    def __init__(
            self,
            model: OriginalGANModel,
            mean: List[float],
            std: List[float],
            noise_channel: int,
    ):
        super().__init__(
            model,
            mean,
            std
        )
        self.noise_channel = noise_channel

    def show_generate_results(
            self,
            saved_dir: str,
            desc: str = 'show generate result',
            generate_num: int = 64,
            *args,
            **kwargs
    ):
        print()
        print(desc)
        print()

        os.makedirs(saved_dir, exist_ok=True)

        self.model.generator.eval()
        self.model.discriminator.eval()

        fake_images: torch.Tensor = self.model.get_fake_images(
            generate_num,
            self.noise_channel
        )
        n, c, h, w = fake_images.shape

        if c == 1:
            """
            Gray to RGB(or BGR)
            """
            fake_images = fake_images.expand(
                size=(n, 3, h, w)
            )

        for image_index in range(fake_images.shape[0]):
            fake_img_np = OriginalGANTool.image_tensor_to_np(
                fake_images[image_index],
                self.mean,
                self.std
            )
            saved_path = '{}/{}_fake.png'.format(saved_dir, image_index)
            CV2.imwrite(saved_path, fake_img_np)
