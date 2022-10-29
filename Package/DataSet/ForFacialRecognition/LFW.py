import os
from torch.utils.data import Dataset, DataLoader
import albumentations as alb
from Package.BaseDev import CV2
import torch


class LFWDataSet(Dataset):
    def __init__(
            self,
            root: str,
            transform: alb.Compose,
            train: bool = True
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.train = train
        self.id_vec, self.image_path_to_id = self.get_info()
        self.image_path_vec = list(self.image_path_to_id.keys())

    def get_info(self):
        path = self.root
        id_vec = os.listdir(self.root)
        image_path_to_id = {}
        """
                image_path(abs): image_id

        """
        for name in id_vec:

            image_names = os.listdir(os.path.join(path, name))
            for val in image_names:
                image_path_to_id[os.path.join(path, name, val)] = name
        return id_vec, image_path_to_id

    def __len__(self):
        return len(self.image_path_vec)

    def pull(
            self,
            index
    ):
        img_path = self.image_path_vec[index]
        img = CV2.imread(img_path)

        img_id = self.image_path_to_id.get(img_path)
        if self.train:
            """
            when training,
            we need return img and target.
            target will be used to compute loss!
            """
            return img, self.id_vec.index(img_id)

        else:
            """
            when testing,
            we need return img and image_path.
            image_path will be used to store feature of this image. That's important!
            """
            return img, img_path

    def __getitem__(self, index):

        img, target_or_path = self.pull(index)

        res = self.transform(image=img)
        trans_img = res.get('image')
        trans_img_tensor = torch.tensor(trans_img, dtype=torch.float32)
        trans_img_tensor = trans_img_tensor.permute(2, 0, 1)
        return trans_img_tensor, target_or_path


def get_data_pair(
    pair_txt: str,
    img_path: str
):
    path = img_path
    with open(pair_txt, mode='r') as f:
        lines = f.readlines()
        data_pair = []
        """
            item:
                [image_a_path, image_b_path, is_same_id]
        """
        for ind in range(1, len(lines)):
            tmp = lines[ind][:-1].split('\t')
            if len(tmp) == 3:
                image_a_path = os.path.join(
                    path,
                    tmp[0],
                    '{}_{}.jpg'.format(
                        tmp[0],
                        '0' * (4 - len(tmp[1])) + tmp[1]
                    ))
                image_b_path = os.path.join(
                    path,
                    tmp[0],
                    '{}_{}.jpg'.format(
                        tmp[0],
                        '0' * (4 - len(tmp[2])) + tmp[2]
                    ))
                data_pair.append([
                    image_a_path,
                    image_b_path,
                    1
                ])
            else:
                image_a_path = os.path.join(
                    path,
                    tmp[0],
                    '{}_{}.jpg'.format(
                        tmp[0],
                        '0' * (4 - len(tmp[1])) + tmp[1]
                    ))
                image_b_path = os.path.join(
                    path,
                    tmp[2],
                    '{}_{}.jpg'.format(
                        tmp[2],
                        '0' * (4 - len(tmp[3])) + tmp[3]
                    ))
                data_pair.append([
                    image_a_path,
                    image_b_path,
                    0
                ])
    return data_pair


def get_lfw_loader(
        root: str,
        train: bool = True,
        trans: alb.Compose = None,
        batch_size: int = 128,
        num_workers: int = 8
):
    lfw = LFWDataSet(
        root,
        trans,
        train=train
    )
    shuffle = train
    loader = DataLoader(lfw, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    root = '/home/dell/data/DataSet/LFWoriginal/images'
    pair_path = '/home/dell/data/DataSet/LFWoriginal/pairs.txt'
    trans = alb.Compose([
        alb.Resize(112, 112),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    da_loader = get_lfw_loader(
        root,
        True,
        trans,
        128,
        8
    )

