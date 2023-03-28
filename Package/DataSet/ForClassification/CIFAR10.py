from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch


def get_cifar_10_data_loader(
        root: str,
        train: bool,
        trans: transforms.Compose,
        batch_size: int,
        num_workers: int
) -> DataLoader:
    data_set = CIFAR10(
        root=root,
        train=train,
        transform=trans
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )
    return data_loader


class PartCIFAR10(Dataset):
    def __init__(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            trans: transforms.Compose,
    ):
        super().__init__()
        self.images = images
        self.labels = labels
        self.trans = trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.trans(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def get_part_cifar10_data_loader(
        root: str,
        train: bool,
        kind_num: int,
        trans: transforms.Compose,
        batch_size: int,
        num_workers: int
):
    all_kind_data_set = CIFAR10(
        root=root,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    all_kind_data_loader = DataLoader(
        all_kind_data_set,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )
    x_vec = np.empty(shape=(0, 3, 32, 32), dtype=np.uint8)
    y_vec = np.empty(shape=(0,), dtype=np.int32)

    for _, data in enumerate(all_kind_data_loader):
        x: torch.Tensor = data[0]
        y: torch.Tensor = data[1]
        mask = y < kind_num
        x = x[mask]
        y = y[mask]
        if x.shape[0] != 0:
            x_vec = np.concatenate([x_vec, (x.numpy() * 255.0).astype(np.uint8)], axis=0)
            y_vec = np.concatenate([y_vec, y.numpy()], axis=0)

    x_vec = np.transpose(x_vec, axes=(0, 2, 3, 1))
    data_set = PartCIFAR10(
        x_vec,
        y_vec,
        trans
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )
    return data_loader


if __name__ == "__main__":

    loader = get_part_cifar10_data_loader(
        '/home/dell/data/DataSet/cifar10/data',
        True,
        kind_num=4,
        trans=transforms.Compose([
            transforms.ToTensor(),
            ]),
        batch_size=128,
        num_workers=8
    )
    for _, (x, y) in enumerate(loader):
        print(x.shape, y)
