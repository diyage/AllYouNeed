"""
this file is used for create data set/loader(for segmentation tasks)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as alb
from typing import List
import os
import xml.etree.ElementTree as ET
from Package.BaseDev.cv2_ import CV2


class VocDataSetForAllTasks(Dataset):
    def __init__(
            self,
            root: str,
            years: list,
            train: bool,
            image_size: int,
            kind_name_to_color: dict,
            mean: list = [0.485, 0.456, 0.406],
            std: list = [0.229, 0.224, 0.225],
            transform: alb.Compose = None,
            use_bbox: bool = True,
            use_mask_type: int = 0,

    ):
        super().__init__()
        self.root = root
        self.years = years
        self.train = train
        self.KIND_NAME_TO_COLOR = kind_name_to_color

        if self.train:
            self.data_type = 'trainval'
        else:
            self.data_type = 'test'
            self.years = ['2007']
            print('Just VOC2007 has test images/masks(Segmentation)!'.center(100, '*'))
        self.image_size = image_size

        if transform is None:
            self.transform = alb.Compose([
                alb.Resize(image_size, image_size),
                alb.Normalize(mean, std)
            ], bbox_params=alb.BboxParams(format='pascal_voc'))
        else:
            self.transform = transform

        assert use_mask_type in [-1, 0, 1]
        self.use_mask_type = use_mask_type
        # -1 mask for semantic, 0 do not use mask, 1 mask for instance
        self.use_bbox = use_bbox

        self.images_objects_masks_path = self.__get_all_path()

    def set_use_bbox(
            self,
            use_bbox: bool
    ):
        self.use_bbox = use_bbox

    def set_use_mask_type(
            self,
            mask_type: int
    ):
        assert mask_type in [-1, 0, 1]
        self.use_mask_type = mask_type
        self.images_objects_masks_path = self.__get_all_path()

    def __get_all_path(
            self
    ) -> List:
        res = []

        for year in self.years:
            images_path = os.path.join(
                self.root,
                year,
                self.data_type,
                'JPEGImages',
            )
            objects_path = os.path.join(
                self.root,
                year,
                self.data_type,
                'Annotations',
            )
            masks_path = os.path.join(
                self.root,
                year,
                self.data_type,
                'SegmentationObject' if self.use_mask_type == 1 else 'SegmentationClass',
            )
            if self.use_mask_type == 0:
                names_vec = os.listdir(objects_path)
            else:
                names_vec = os.listdir(masks_path)
            res += [
                (
                    os.path.join(images_path, name[:-3]+'jpg'),
                    os.path.join(objects_path, name[:-3] + 'xml'),
                    os.path.join(masks_path, name[:-3]+'png'),
                ) for name in names_vec
            ]

        return res

    @staticmethod
    def read_xml_objects(
            xml_file_name: str
    ) -> List[List]:
        position_kind_name_vec = []
        root = ET.parse(xml_file_name).getroot()  # type: xml.etree.ElementTree.Element
        for obj in root.iter('object'):
            kind = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            a = float(bbox.find('xmin').text.strip())  # point to x_axis dist
            b = float(bbox.find('ymin').text.strip())  # point to y_axis dist
            m = float(bbox.find('xmax').text.strip())
            n = float(bbox.find('ymax').text.strip())
            position_kind_name_vec.append([a, b, m, n, kind])
        return position_kind_name_vec

    def split_mask(
            self,
            mask_path: str
    ) -> List[np.ndarray]:
        """
        read the mask_path of an image.
        then, split this mask(it is also an image) to one mask_vec(
        save many masks, each item is one object or class).

        """
        mask = CV2.imread(mask_path).astype(np.int32)
        # print(mask_path)
        # CV2.imshow('origin_mask', mask.astype(np.uint8))
        # CV2.waitKey(0)
        if self.use_mask_type == 1:
            """
            for instance segmentation,
            each instance(object) has its own color(and only one).
            """
            res = []
            mask_sum = mask[..., 0] * 1000000 + mask[..., 1] * 1000 + mask[..., 2] * 1
            a = np.reshape(mask_sum, newshape=(-1,)).tolist()
            b = list(set(tuple(a)))
            for val in b:
                if val == 192224224 or val == 0:
                    # color (192, 224, 224)  --> bound line (ignore)
                    # color (0, 0, 0)  --> back_ground (ignore)
                    continue
                val_str = str(val)
                now_obj_color_sum_str = '0' * (9 - len(val_str)) + val_str
                now_obj_color = (
                    int(now_obj_color_sum_str[0:3]),
                    int(now_obj_color_sum_str[3:6]),
                    int(now_obj_color_sum_str[6:9])
                )
                now_obj_color = np.array(now_obj_color, dtype=np.uint8)
                temp = (mask == now_obj_color).astype(np.float32).sum(-1)  # (H, W)
                now_obj_mask = (temp == 3).astype(np.float32)
                res.append(now_obj_mask)
            return res
        else:
            """
            for semantic segmentation,
            each class has its own color(and only one).
            and will return one mask_vec(has class_num + 1 masks)
            """
            res = []
            for kind_name, color in self.KIND_NAME_TO_COLOR.items():
                kind_mask = ((mask - np.array(color)) == 0).astype(np.float32)  # type: np.ndarray
                kind_mask = kind_mask.sum(axis=2)
                kind_mask = (kind_mask == 3).astype(np.float32)  # (H, W)  0.0/1.0
                res.append(kind_mask)
            return res

    def pull_an_image(
            self,
            index: int
    ) -> List:
        res = []
        image_path, obj_path, mask_path = self.images_objects_masks_path[index]

        image = CV2.imread(image_path)
        res.append(image)

        if self.use_bbox:
            objects_vec = self.read_xml_objects(obj_path)
            res.append(objects_vec)
        else:
            res.append([])

        if self.use_mask_type != 0:
            masks_vec = self.split_mask(mask_path)
            res.append(masks_vec)
        else:
            res.append([])

        return res

    def __len__(self):
        return len(self.images_objects_masks_path)

    def __getitem__(
            self,
            index
    ):
        image, objects_vec, masks_vec = self.pull_an_image(index)
        res = self.transform(image=image, bboxes=objects_vec, masks=masks_vec)
        new_image = res.get('image')
        new_obj_vec = res.get('bboxes')
        new_mask_vec = res.get('masks')
        # if len(new_obj_vec) != len(new_mask_vec):
        #     print('wow')
        if self.use_mask_type == 0:
            new_mask_vec = np.zeros(
                shape=(len(list(self.KIND_NAME_TO_COLOR.keys())), new_image.shape[0], new_image.shape[1])
            )
        """
can not return mask(torch.Tensor type) when instance segmentation task.
cause dimensions are not equal.
        """
        return torch.tensor(new_image, dtype=torch.float32).permute(2, 0, 1), new_obj_vec, new_mask_vec

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        imgs = batch[0]
        objects = batch[1]
        masks = batch[2]
        del batch
        return torch.stack(imgs), objects, masks


def get_voc_for_all_tasks_loader(
        root: str,
        years: List[str],
        train: bool,
        image_size: int,
        kind_name_to_color: dict,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        trans_form: alb.Compose = None,
        batch_size: int = 8,
        num_workers: int = 4,
        use_bbox: bool = True,
        use_mask_type: int = -1
) -> DataLoader:
    data_set = VocDataSetForAllTasks(
        root,
        years,
        train,
        image_size,
        kind_name_to_color,
        mean,
        std,
        trans_form,
        use_bbox=use_bbox,
        use_mask_type=use_mask_type
    )

    data_loader = DataLoader(
        data_set,
        shuffle=True if train else False,
        batch_size=batch_size,
        collate_fn=data_set.collate_fn,
        num_workers=num_workers
    )
    return data_loader


def debug_for_show(
        img: np.ndarray,
        obj_vec: List[List],
        mask_vec: List[np.ndarray],
        pre_fix: str = ''
):
    for obj in obj_vec:
        CV2.rectangle(
            img,
            start_point=(int(obj[0]), int(obj[1])),
            end_point=(int(obj[2]), int(obj[3])),
            color=(0, 0, 0),
            thickness=2
        )
    CV2.imshow(pre_fix+'image', img)
    CV2.waitKey(0)
    i = 0
    for mask in mask_vec:
        t = np.expand_dims(mask*255.0, -1).repeat(3, -1).astype(np.uint8)
        CV2.imshow(pre_fix+'mask:{}'.format(i), t)
        CV2.waitKey(0)
        i += 1


def debug_pull_an_image():
    d = VocDataSetForAllTasks(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        416,
        KIND_NAME_TO_COLOR,
        use_bbox=True,
        use_mask_type=-1
    )
    # for semantic
    img, obj_vec, mask_vec = d.pull_an_image(index=np.random.randint(0, len(d)))
    debug_for_show(img, obj_vec, mask_vec, pre_fix='semantic_')
    # for instance
    d.set_use_mask_type(1)
    img, obj_vec, mask_vec = d.pull_an_image(index=np.random.randint(0, len(d)))
    debug_for_show(img, obj_vec, mask_vec, pre_fix='instance_')


def debug_transform():
    trans = alb.Compose([
        alb.Rotate(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(),
        alb.HueSaturationValue(),
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    d = VocDataSetForAllTasks(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        416,
        KIND_NAME_TO_COLOR,
        transform=trans,
        use_bbox=True,
        use_mask_type=1,
    )

    img, obj_vec, mask_vec = d.pull_an_image(index=np.random.randint(0, len(d)))
    debug_for_show(img.copy(), obj_vec, mask_vec, pre_fix='old --> ')
    res = trans(image=img, bboxes=obj_vec, masks=mask_vec)
    new_image = res.get('image')
    new_obj_vec = res.get('bboxes')
    new_mask_vec = res.get('masks')
    debug_for_show(new_image.copy(), new_obj_vec, new_mask_vec, pre_fix='new --> ')


def debug_data_set():
    trans = alb.Compose([
        alb.Rotate(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(),
        alb.HueSaturationValue(),
        alb.Resize(416, 416),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    data_set = VocDataSetForAllTasks(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        416,
        KIND_NAME_TO_COLOR,
        transform=trans,
        use_bbox=False,
        use_mask_type=0
    )
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        data_set,
        shuffle=True,
        batch_size=8,
        collate_fn=data_set.collate_fn
    )
    print('use_bbox: {}, use_mask_type: {}(please make sure you know what mean it is)'.format(
        data_set.use_bbox,
        data_set.use_mask_type
    ))
    for _, (images, objects, masks) in enumerate(data_loader):
        print(images.shape)
        for i in range(images.shape[0]):
            print(len(objects[i]), len(masks[i]))
        break


if __name__ == '__main__':
    """
        color of mask(for class)
    """
    KIND_NAME_TO_COLOR = {
        'background': (0, 0, 0),
        'aeroplane': (0, 0, 128),
        'bicycle': (0, 128, 0),
        'bird': (0, 128, 128),
        'boat': (128, 0, 0),
        'bottle': (128, 0, 128),
        'bus': (128, 128, 0),
        'car': (128, 128, 128),
        'cat': (0, 0, 64),
        'chair': (0, 0, 192),
        'cow': (0, 128, 64),
        'dining table': (0, 128, 192),
        'dog': (128, 0, 64),
        'horse': (128, 0, 192),
        'motorbike': (128, 128, 64),
        'person': (128, 128, 192),
        'potted plant': (0, 64, 0),
        'sheep': (0, 64, 128),
        'sofa': (0, 192, 0),
        'train': (0, 192, 128),
        'tv monitor': (128, 64, 0),
        # 'bordering region': (192, 224, 224),
    }

    # debug_pull_an_image()
    # debug_transform()
    debug_data_set()

