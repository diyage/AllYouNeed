from torch.utils.data import DataLoader, Dataset
import albumentations as alb
from Package.BaseDev import CV2
import os
import numpy as np
import torch


class Portrait(Dataset):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: alb.Compose,
            strict_pair: bool = True,
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.strict_pair = strict_pair
        self.a_style_path_vec, self.b_style_path_vec = self.get_info()

        if not self.strict_pair:
            self.a_style_path_vec = np.random.choice(
                self.a_style_path_vec,
                size=(len(self.a_style_path_vec), ),
                replace=False
            ).tolist()
            self.b_style_path_vec = np.random.choice(
                self.b_style_path_vec,
                size=(len(self.b_style_path_vec), ),
                replace=False
            ).tolist()

    def get_info(self):
        if self.train:
            prefix = 'train'
        else:
            prefix = 'test'
        a_style_dir = os.path.join(
            self.root,
            prefix+'A'
        )
        b_style_dir = os.path.join(
            self.root,
            prefix+'B'
        )
        a_style_image_name = os.listdir(a_style_dir)
        b_style_image_name = [
            'B' + a_image_name[1:] for a_image_name in a_style_image_name
        ]
        a_style_path = [
            os.path.join(
                a_style_dir,
                a_image_name
            ) for a_image_name in a_style_image_name
        ]
        b_style_path = [
            os.path.join(
                b_style_dir,
                b_image_name
            ) for b_image_name in b_style_image_name
        ]
        return a_style_path, b_style_path

    def pull_two_images(
            self,
            index: int
    ):
        image_a_path = self.a_style_path_vec[index]
        image_b_path = self.b_style_path_vec[index]
        img_a = CV2.imread(image_a_path)
        img_b = CV2.imread(image_b_path)
        return img_a, img_b

    def __len__(self):
        return len(self.a_style_path_vec)

    def __getitem__(self, index):
        a, b = self.pull_two_images(index)

        trans_a = self.transform(image=a).get('image')
        trans_b = self.transform(image=b).get('image')

        a_tensor = torch.tensor(trans_a, dtype=torch.float32).permute(2, 0, 1)
        b_tensor = torch.tensor(trans_b, dtype=torch.float32).permute(2, 0, 1)
        return a_tensor, b_tensor


def get_portrait_data_loader(
        root: str,
        train: bool,
        transform: alb.Compose,
        strict_pair: bool,
        batch_size: int,
        num_workers: int = 0,
) -> DataLoader:
    data_set = Portrait(
        root,
        train,
        transform,
        strict_pair=strict_pair,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )
    return data_loader


def debug_data_loader():
    path = '/home/dell/data/DataSet/Portrait'
    image_size_tuple = (256, 256)
    strict_pair = False
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    trans = alb.Compose([
        alb.Resize(*image_size_tuple),
        alb.Normalize(
            mean=mean,
            std=std
        )
    ])

    data_set = Portrait(
        path,
        train=True,
        transform=trans,
        strict_pair=strict_pair
    )

    a, b = data_set.pull_two_images(100)
    CV2.imshow('a', a)
    CV2.waitKey(0)
    CV2.imshow('b', b)
    CV2.waitKey(0)
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=True,
    )
    for _, (a, b) in enumerate(data_loader):
        print(a.shape, b.shape)


if __name__ == "__main__":
    debug_data_loader()
