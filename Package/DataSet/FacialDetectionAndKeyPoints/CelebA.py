from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
import numpy as np


class CelebA(Dataset):
    def __init__(
            self
    ):
        pass

    @staticmethod
    def get_all_original_info(
            root: str
    ) -> dict:
        """

        :param root:
        :return:
        """
        info_name = os.path.join(
            root,
            'info.json'
        )
        if os.path.exists(info_name):
            with open(info_name, mode='r') as f:
                original_info = json.load(f)

            print('\nOriginal info.json exists(from CelebA)! So we do not process again!')

        else:
            original_info = {
                'position': {},
                'key_point': {}
            }
            annotation_path = os.path.join(
                root,
                'Anno'
            )
            image_path = os.path.join(
                root,
                'Img',
                'img_celeba'
            )
            # split_partition_txt_path = os.path.join(
            #     root,
            #     'Eval',
            #     'list_eval_partition.txt'
            # )
            key_point_txt_path = os.path.join(
                annotation_path,
                'list_landmarks_celeba.txt'
            )
            position_txt_path = os.path.join(
                annotation_path,
                'list_bbox_celeba.txt'
            )
            """
            enum
            """
            f_box = open(position_txt_path)
            f_key_point = open(key_point_txt_path)

            for i, (f_box_line, f_landmarks_line) in enumerate(tqdm(
                    zip(f_box, f_key_point),
                    desc='Collect original info from CelebA.',
                    position=0
            )):
                if i < 2:  # skip the top two lines
                    continue

                image_name_box = f_box_line.strip().split(' ')[0]
                image_name_landmarks = f_landmarks_line.strip().split(' ')[0]
                assert image_name_box == image_name_landmarks

                boxes = f_box_line.strip().split(' ')[1:]
                boxes = list(filter(lambda x: x != '', boxes))
                boxes = np.array(boxes).astype(int)

                landmarks = f_landmarks_line.strip().split(' ')[1:]
                landmarks = list(filter(lambda x: x != '', landmarks))
                landmarks = np.array(landmarks).astype(int)

                now_image_abs_path = os.path.join(
                    image_path,
                    image_name_box
                )
                original_info['position'][now_image_abs_path] = boxes.tolist()
                original_info['key_point'][now_image_abs_path] = landmarks.tolist()

            f_box.close()
            f_key_point.close()

            with open(info_name, mode='w') as f:
                json.dump(original_info, f)

            print('\nSave original info.json successfully!')

        return original_info

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
