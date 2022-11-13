from torch.utils.data import Dataset, DataLoader
import json
import albumentations as alb
import typing
import cv2
import numpy as np
import torch


class MTCNNDataSet(Dataset):
    def __init__(
            self,
            cropped_celeb_a_root: str,
            cropped_wider_face_root: str,
            train: bool,
            transform: alb.Compose,
    ):
        super().__init__()
        self.cropped_celeb_a_root = cropped_celeb_a_root
        self.cropped_wider_face_root = cropped_wider_face_root
        self.train = train
        self.transform = transform
        self.celeb_a_info, self.wider_face_info = self.get_info()

        self.landmark_keys, self.positive_position_keys, \
            self.negative_position_keys, self.part_position_keys = self.get_used_keys()
        self.len = len(self.landmark_keys)

    def get_used_keys(self):
        train_or_test = 'train' if self.train else 'test'

        with open('{}/{}.json'.format(self.cropped_celeb_a_root, train_or_test), mode='r') as f:
            landmark_keys: typing.List[str] = json.load(f)

        with open('{}/{}.json'.format(self.cropped_wider_face_root, train_or_test), mode='r') as f:
            position_keys: dict = json.load(f)
            positive_position_keys: typing.List[str] = position_keys['positive']
            negative_position_keys: typing.List[str] = position_keys['negative']
            part_position_keys: typing.List[str] = position_keys['part']

        min_len = min(
            len(landmark_keys),
            len(positive_position_keys),
            int(1.0 * len(negative_position_keys)/3),
            len(part_position_keys)
        )

        # if min_len >= 50000:
        #     print(">>> The original max length could be set {}, it is too big! ".format(min_len))
        #     min_len = 50000

        print("\n>>> We keep landmark:positive:negative:part --> 1:1:3:1 (length is {})".format(min_len))

        return landmark_keys[: min_len], positive_position_keys[:min_len], \
            negative_position_keys[: int(3*min_len)], part_position_keys[:min_len]

    def get_info(
            self
    ) -> typing.Tuple[dict, dict]:
        with open('{}/info.json'.format(self.cropped_celeb_a_root), mode='r') as f:
            celeb_a_info: dict = json.load(f)
        with open('{}/info.json'.format(self.cropped_wider_face_root), mode='r') as f:
            wider_face_info: dict = json.load(f)

        return celeb_a_info, wider_face_info

    def __len__(self):
        return self.len

    def pull_image_and_landmark_offset(
            self,
            index: int
    ):
        img_abs_path = self.landmark_keys[index]
        image = cv2.imread(img_abs_path)
        key_point = self.celeb_a_info['key_point'][img_abs_path]

        return image, key_point

    def pull_positive_image_and_position_offset(
            self,
            index: int
    ):
        img_abs_path = self.positive_position_keys[index]
        image = cv2.imread(img_abs_path)
        position = self.wider_face_info['position'][img_abs_path]
        return image, position

    def pull_negative_images(
            self,
            index: int
    ):
        res = []
        for i in range(3):
            real_ind = index + int(i * self.len)
            img_abs_path = self.negative_position_keys[real_ind]
            image = cv2.imread(img_abs_path)
            res.append(image)
        return res

    def pull_part_image_and_position_offset(
            self,
            index: int
    ):
        img_abs_path = self.part_position_keys[index]
        image = cv2.imread(img_abs_path)
        position = self.wider_face_info['position'][img_abs_path]
        return image, position

    def __getitem__(self, index):
        img_for_key_point, key_point_offset = self.pull_image_and_landmark_offset(index)
        img_for_positive, positive = self.pull_positive_image_and_position_offset(index)
        img_for_negative_vec = self.pull_negative_images(index)
        img_for_part, part = self.pull_part_image_and_position_offset(index)
        img_vec = [
            img_for_key_point,  # 0
            img_for_positive,  # 1
            *img_for_negative_vec,  # 2, 3, 4
            img_for_part  # 5
        ]
        trans_img_vec = []
        for img in img_vec:
            res = self.transform(image=img)
            trans_img: np.ndarray = res.get('image')
            trans_img_vec.append(trans_img)

        img_tensor = torch.tensor(
            np.array(trans_img_vec),
            dtype=torch.float32
        )
        """
        from shape: [6, h, w, 3]
        """
        img_tensor = img_tensor.permute(3, 1, 2, 0)
        """
        to shape: [3, h, w, 6]
        """

        key_point_offset_tensor = torch.tensor(
            np.array(key_point_offset),
            dtype=torch.float32
        )
        """
            shape: [10, ]
        """

        positive_tensor = torch.tensor(
            np.array(positive),
            dtype=torch.float32
        )
        """
        shape: [4, ]
        """

        part_tensor = torch.tensor(
            np.array(part),
            dtype=torch.float32
        )
        """
        shape: [4, ]
        """
        return img_tensor, key_point_offset_tensor, positive_tensor, part_tensor


def get_mt_cnn_data_loader(

):
    pass
