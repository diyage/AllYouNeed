from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def get_cifar_100_data_loader(
        root: str,
        train: bool,
        trans: transforms.Compose,
        batch_size: int,
        num_workers: int
) -> DataLoader:
    data_set = CIFAR100(
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


if __name__ == "__main__":

    loader = get_cifar_100_data_loader(
        '/home/dell/data/DataSet/cifar100/data',
        True,
        trans=transforms.Compose([
            transforms.ToTensor(),
            ]),
        batch_size=128,
        num_workers=8
    )
    for _, (x, y) in enumerate(loader):
        print(x.shape)
