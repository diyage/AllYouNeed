from torch.utils.data import Dataset, DataLoader
import albumentations as alb
import os
import numpy as np
from Package.BaseDev import CV2
import torch


class MS1M(Dataset):
    def __init__(
            self,
            root: str,
            transform: alb.Compose,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_name_path_and_id = self.get_info()

    def get_info(self):
        path = self.root
        id_vec = os.listdir(path)
        print()
        print('MS1M class_num(total id): {}'.format(len(id_vec)))
        print()
        res = []
        for now_id in id_vec:
            image_name_vec = os.listdir("{}/{}".format(path, now_id))
            res += [
                (
                    os.path.join(path, now_id, now_name),
                    int(now_id)
                ) for now_name in image_name_vec
            ]
        return res

    def __len__(self):
        return len(self.image_name_path_and_id)

    def __getitem__(self, index):
        img_name_path, now_id = self.image_name_path_and_id[index]
        img: np.ndarray = CV2.imread(img_name_path)

        res = self.transform(image=img)
        trans_img = res.get('image')
        trans_img_tensor = torch.tensor(trans_img, dtype=torch.float32)
        trans_img_tensor = trans_img_tensor.permute(2, 0, 1)

        trans_id_tensor = torch.tensor(now_id, dtype=torch.long)

        return trans_img_tensor, trans_id_tensor


def get_ms1m_data_loader(
        root: str,
        trans: alb.Compose,
        batch_size: int = 128,
        num_workers: int = 8
) -> DataLoader:
    ms1m = MS1M(
        root,
        trans
    )
    train_loader = DataLoader(
        ms1m,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_loader


def debug_ms1m_loader():
    ms1m_train_loader = get_ms1m_data_loader(
        root='/home/dell/data/DataSet/faces_ms1m_112x112/images',
        trans=alb.Compose([
            alb.Resize(112, 112),
            alb.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )]),
        batch_size=128,
        num_workers=0
    )
    for _, (img, target) in enumerate(ms1m_train_loader):
        print(img.shape)
        print(target.shape)
        break


if __name__ == '__main__':
    debug_ms1m_loader()
