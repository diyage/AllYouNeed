from torch.utils.data import Dataset, DataLoader
from typing import List
import albumentations as alb
import json
import cv2
import torch
import numpy as np


KINDS_NAME = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}


class COCODataSet(Dataset):
    def __init__(
            self,
            root: str,
            years: List[str],
            train: bool,
            image_size: int,
            transform: alb.Compose,
            use_mosaic: bool = True,
    ):
        super().__init__()
        self.root = root
        self.years = years
        self.train = train
        self.image_size = image_size
        self.transform = transform
        self.info = self.get_info()
        self.all_index = list(range(len(self.info)))
        self.use_mosaic = use_mosaic
        print('\nUse COCO --> years:{} | train: {} | total images num:{}\n'.format(years, train, len(self.info)))
        self.img_cache = [None]*len(self.info)
        """
        img_cache used for decreasing time of loading data
        """

    def pull_an_image_label(
            self,
            index: int,
    ):
        img_path, bbox_kind_id_list = self.info[index]

        img = cv2.imread(img_path)

        return img, bbox_kind_id_list

    def trans_an_image_label(
            self,
            image: np.ndarray,
            objects_vec: list,
    ):
        res = self.transform(image=image, bboxes=objects_vec)

        new_image = res.get('image')
        new_obj_vec = res.get('bboxes')

        new_obj_vec = [[KINDS_NAME[obj[-1]], *(obj[:-1])] for obj in new_obj_vec]

        return new_image, new_obj_vec

    def get_normal_image_label(
            self,
            index: int
    ):
        image, objects_vec = self.pull_an_image_label(index)

        new_image, new_obj_vec = self.trans_an_image_label(image, objects_vec)

        return new_image, new_obj_vec

    def get_mosaic_image_label(
            self,
            index: int
    ):
        rand_int = np.random.randint(low=0, high=2)
        if rand_int == 0:
            image, objects_vec = self.pull_mosaic_image_label(index)
        else:
            image, objects_vec = self.pull_an_image_label(index)

        new_image, new_obj_vec = self.trans_an_image_label(image, objects_vec)

        return new_image, new_obj_vec

    def pull_mosaic_image_label(
            self,
            index: int
    ):

        other_index = self.all_index[:index] + self.all_index[index+1:]
        all_index = [index] + np.random.choice(other_index, size=(3, ), replace=False).tolist()

        all_images = []  # (4, ...)
        all_bbox_kind_id_list = []  # (4, ...)
        max_h = 0
        max_w = 0
        for r in range(2):
            for c in range(2):
                img_pos = r * 2 + c
                now_index = all_index[img_pos]
                img, bbox_kind_id_list = self.pull_an_image_label(now_index)

                max_h = max(max_h, img.shape[0])
                max_w = max(max_w, img.shape[1])

                all_images.append(img)
                all_bbox_kind_id_list.append(bbox_kind_id_list)

        complex_images = np.zeros(shape=(max_h * 2, max_w * 2, 3), dtype=np.uint8)
        complex_bbox_kind_id_list = []

        for r in range(2):
            for c in range(2):
                img_pos = r * 2 + c

                img = all_images[img_pos]
                bbox_kind_id_list = all_bbox_kind_id_list[img_pos]

                img_h, img_w, _ = img.shape

                shift_h = r * max_h + int(np.random.random() * (max_h - img_h))
                shift_w = c * max_w + int(np.random.random() * (max_w - img_w))

                complex_images[shift_h: shift_h+img_h, shift_w: shift_w+img_w, :] = img

                for bbox_kind_id in bbox_kind_id_list:
                    complex_bbox_kind_id_list.append([
                        bbox_kind_id[0] + shift_w,  # x1
                        bbox_kind_id[1] + shift_h,  # y1
                        bbox_kind_id[2] + shift_w,  # x2
                        bbox_kind_id[3] + shift_h,  # y2
                        bbox_kind_id[4],  # kind id
                    ])

        return complex_images, complex_bbox_kind_id_list

    def get_info(
            self
    ) -> List:

        image_id_to_path_and_bbox_kind_id = {}
        train_or_val = 'train' if self.train else 'val'
        for year in self.years:
            json_file_name = "{}/{}/annotations_trainval{}/annotations/instances_{}{}.json".format(
                self.root,
                year,
                year,
                train_or_val,
                year
            )
            with open(json_file_name, mode='r') as f:
                json_file = json.load(f)

                temp = json_file['images']
                for index in range(len(temp)):
                    image_id = str(temp[index]['id'])

                    file_name = temp[index]['file_name']

                    if image_id_to_path_and_bbox_kind_id.get(image_id) is None:

                        image_id_to_path_and_bbox_kind_id[image_id] = [
                            "{}/{}/{}{}/{}".format(self.root, year, train_or_val, year, file_name),
                            []  # used for bbox_kind_id
                        ]

                temp = json_file['annotations']
                for index in range(len(temp)):
                    x, y, w, h = temp[index]['bbox']

                    if w < 1 or h < 1:
                        continue

                    now_box_image_id = str(temp[index]['image_id'])

                    now_kind_id = temp[index]['category_id']
                    image_id_to_path_and_bbox_kind_id[now_box_image_id][1].append(
                        [
                            x, y, x+w, y+h, now_kind_id
                        ]
                    )

        return list(image_id_to_path_and_bbox_kind_id.values())

    def __len__(self):
        return len(self.info)

    def __getitem__(
            self,
            index
    ):
        if self.use_mosaic:
            new_image, new_obj_vec = self.get_mosaic_image_label(index)
        else:
            new_image, new_obj_vec = self.get_normal_image_label(index)

        return torch.tensor(new_image, dtype=torch.float32).permute(2, 0, 1), new_obj_vec

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        imgs = batch[0]
        objects = batch[1]

        del batch
        return torch.stack(imgs), objects


def get_coco_data_loader(
        root: str,
        years: List[str],
        train: bool,
        image_size: int,
        trans_form: alb.Compose = None,
        batch_size: int = 8,
        num_workers: int = 4,
        use_mosaic: bool = True
):
    data_set = COCODataSet(
        root,
        years,
        train,
        image_size,
        trans_form,
        use_mosaic
    )

    data_loader = DataLoader(
        data_set,
        shuffle=True if train else False,
        batch_size=batch_size,
        collate_fn=data_set.collate_fn,
        num_workers=num_workers
    )
    return data_loader


def debug_pull_img():
    trans = alb.Compose([
        alb.GaussNoise(),
        # alb.HueSaturationValue(),
        # alb.RandomBrightnessContrast(),
        # alb.ColorJitter(),
        # alb.Blur(3),
        alb.HorizontalFlip(),
        alb.Rotate(limit=(-30, 30), p=1.0),
        # alb.RandomResizedCrop(
        #     416,
        #     416,
        #     p=1.0,
        #     scale=(0.75, 1.0)
        # ),

        # alb.GaussNoise(var_limit=(60, 150), p=1),

    ], bbox_params=alb.BboxParams(format='pascal_voc'))
    coco = COCODataSet(
        '/home/dell/data/DataSet/COCO/',
        years=['2014'],
        train=True,
        image_size=608,
        transform=trans
    )
    img, bbox_kind_id_list = coco.pull_an_image_label(300)
    print(bbox_kind_id_list)
    res = trans(image=img, bboxes=bbox_kind_id_list)
    img, bbox_kind_id_list = res.get('image'), res.get('bboxes')
    print(bbox_kind_id_list)
    for bbox_kind_id in bbox_kind_id_list:
        x1, y1, x2, y2, kind_id = bbox_kind_id
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 0), thickness=3)
    cv2.imshow('', img)
    cv2.waitKey(0)


def debug_pull_mosaic():
    trans = alb.Compose([
        alb.GaussNoise(),
        # alb.HueSaturationValue(),
        # alb.RandomBrightnessContrast(),
        # alb.ColorJitter(),
        # alb.Blur(3),
        alb.HorizontalFlip(),
        # alb.Rotate(limit=(-20, 20), p=1.0),
        # alb.RandomResizedCrop(
        #     416,
        #     416,
        #     p=1.0,
        #     scale=(0.75, 1.0)
        # ),

        # alb.GaussNoise(var_limit=(60, 150), p=1),

    ], bbox_params=alb.BboxParams(format='pascal_voc'))
    coco = COCODataSet(
        '/home/dell/data/DataSet/COCO/',
        years=['2014'],
        train=True,
        image_size=608,
        transform=trans
    )
    img, bbox_kind_id_list = coco.pull_mosaic_image_label(300)
    print(bbox_kind_id_list)
    res = trans(image=img, bboxes=bbox_kind_id_list)
    img, bbox_kind_id_list = res.get('image'), res.get('bboxes')
    print(bbox_kind_id_list)
    for bbox_kind_id in bbox_kind_id_list:
        x1, y1, x2, y2, kind_id = bbox_kind_id
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 0), thickness=3)
    cv2.imshow('', img)
    cv2.waitKey(0)


def debug_loader():
    root = '/home/dell/data/DataSet/COCO/'
    years = ['2014', '2017']
    train = True
    image_size = 608
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    trans = alb.Compose([
        alb.GaussNoise(),
        # alb.HueSaturationValue(),
        # alb.RandomBrightnessContrast(),
        # alb.ColorJitter(),
        # alb.Blur(3),
        alb.HorizontalFlip(),
        alb.Rotate(limit=(-30, 30), p=1.0),
        # alb.RandomResizedCrop(
        #     416,
        #     416,
        #     p=1.0,
        #     scale=(0.75, 1.0)
        # ),
        alb.Resize(image_size, image_size),
        alb.Normalize(
            mean,
            std
        )
        # alb.GaussNoise(var_limit=(60, 150), p=1),

    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    loader = get_coco_data_loader(
        root,
        years,
        train,
        image_size,
        trans,
        batch_size=16,
        num_workers=4
    )
    # for batch_id, res in enumerate(loader):
    #     img_tensor, label_list = res
    #     # print(label_list)
    #     print(batch_id)


if __name__ == '__main__':
    print("\nDeBUG:\n")
    # debug_loader()
    debug_pull_mosaic()
