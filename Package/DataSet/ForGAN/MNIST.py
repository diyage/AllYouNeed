from torch.utils.data import Dataset, DataLoader
from Package.BaseDev import CV2
import torch
import numpy as np
import os
import albumentations as alb


class MNIST(Dataset):
    def __init__(
            self,
            root: str,
            transform: alb.Compose,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_abs_path = self.get_info()

    def get_info(self):
        path = self.root
        class_name_vec = os.listdir(path)
        res = []
        for class_name in class_name_vec:
            image_dir = os.path.join(path, class_name)
            image_name_vec = os.listdir(image_dir)
            res += [
                os.path.join(image_dir, image_name) for image_name in image_name_vec
            ]
        return res

    def __len__(self):
        return len(self.image_abs_path)

    def pull_an_image(
            self,
            index: int
    ) -> np.ndarray:
        path = self.image_abs_path[index]
        img = CV2.imread(path)
        return img

    def __getitem__(self, index):
        img = self.pull_an_image(index)
        trans_res = self.transform(image=img)
        img_trans = trans_res.get('image')
        """
        img_trans shape: H * W * 3
        """
        img_tensor = torch.tensor(
            img_trans[..., 0],
            dtype=torch.float32
        ).unsqueeze(0)
        return img_tensor


def get_mnist_loader(
        root: str,
        trans: alb.Compose,
        batch_size: int,
        num_workers: int
) -> DataLoader:
    data_set = MNIST(
        root=root,
        transform=trans
    )
    data_loader = DataLoader(
        data_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return data_loader


def debug_loader():
    data_set = MNIST(
        root='/home/dell/data/DataSet/mnist/images',
        transform=alb.Compose([
            alb.Resize(28, 28),
            alb.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    )
    data_loader = DataLoader(
        data_set,
        batch_size=128
    )
    for _, images in enumerate(data_loader):
        print(images.shape)


if __name__ == '__main__':
    debug_loader()
