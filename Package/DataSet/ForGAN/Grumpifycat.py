from torch.utils.data import Dataset, DataLoader
import os
import albumentations as alb
from Package.BaseDev import CV2
import torch


class GrumpCatDataSet(Dataset):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: alb.Compose,
            training_rate: float = 0.8
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.training = train
        self.training_rate = training_rate
        self.images_path = self.__get_images_path()

    def __get_images_path(self):
        def cut_and_split(a: list, b: list):
            number = int(min(len(a), len(b)))
            a_ = a[0: number]
            b_ = b[0: number]
            train_number = int(number * self.training_rate)

            if self.training:
                return a_[0: train_number], b_[0: train_number]
            else:
                return a_[train_number:], b_[train_number:]

        style_a_and_b = os.listdir(self.root)

        style_a_path = os.path.join(self.root, style_a_and_b[0])
        style_b_path = os.path.join(self.root, style_a_and_b[1])

        images_a_names = os.listdir(style_a_path)
        images_b_names = os.listdir(style_b_path)

        images_a_names, images_b_names = cut_and_split(images_a_names, images_b_names)

        res = [
            [os.path.join(style_a_path, image_name) for image_name in images_a_names],
            [os.path.join(style_b_path, image_name) for image_name in images_b_names],
        ]

        return res

    def __len__(self):
        return len(self.images_path[0])

    def pull_two_images(
            self,
            index: int
    ):
        image_a_path, image_b_path = self.images_path[0][index], self.images_path[1][index]

        a = CV2.imread(image_a_path)
        b = CV2.imread(image_b_path)
        return a, b

    def __getitem__(self, index):
        a, b = self.pull_two_images(index)

        trans_a = self.transform(image=a).get('image')
        trans_b = self.transform(image=b).get('image')

        a_tensor = torch.tensor(trans_a, dtype=torch.float32).permute(2, 0, 1)
        b_tensor = torch.tensor(trans_b, dtype=torch.float32).permute(2, 0, 1)
        return a_tensor, b_tensor


def get_grump_cat_data_loader(
        root: str,
        train: bool,
        transform: alb.Compose,
        batch_size: int,
        num_workers: int = 0,
) -> DataLoader:
    data_set = GrumpCatDataSet(
        root,
        train,
        transform,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )
    return data_loader


def debug_data_loader():
    root = '/home/dell/data/DataSet/Grumpifycat'
    image_size_tuple = (256, 256)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    trans = alb.Compose([
        alb.Resize(*image_size_tuple),
        alb.Normalize(
            mean=mean,
            std=std
        )
    ])
    data_set = GrumpCatDataSet(
        root,
        True,
        trans,

    )
    a, b = data_set.pull_two_images(10)
    CV2.imshow('a', a)
    CV2.waitKey(0)
    CV2.imshow('b', b)
    CV2.waitKey(0)
    loader = DataLoader(
        data_set,
        shuffle=True,
        batch_size=1
    )
    for _, (img_a, img_b) in enumerate(loader):
        print(img_a.shape, img_b.shape)


if __name__ == "__main__":
    debug_data_loader()
