import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as alb
from typing import List
import os
import xml.etree.ElementTree as ET
from Package.BaseDev import CV2
from typing import Union, Tuple


class RandomHue(alb.DualTransform):
    def __init__(
            self,
            delta=18.0,
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(always_apply=always_apply, p=p)
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def apply(self, img: np.ndarray, **params):
        if np.random.randint(2):
            img[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img


class RandomSaturation(alb.DualTransform):
    def __init__(
            self,
            lower=0.5,
            upper=1.5,
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def apply(self, img: np.ndarray, **params):
        if np.random.randint(2):
            img[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return img


class ConvertColor(alb.DualTransform):
    def __init__(
            self,
            current='BGR',
            transform='HSV',
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.transform = transform
        self.current = current

    def apply(self, img: np.ndarray, **params):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = CV2.cvtColor(img, CV2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = CV2.cvtColor(img, CV2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(alb.DualTransform):
    def __init__(
            self,
            lower=0.5,
            upper=1.5,
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def apply(self, img: np.ndarray, **params):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            img *= alpha
        return img


class RandomBrightness(alb.DualTransform):
    def __init__(
            self,
            delta=32,
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(always_apply, p)
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def apply(self, img: np.ndarray, **params):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            img += delta
        return img


class PhotometricDistort(alb.DualTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.pd = [
            alb.Compose([
                RandomBrightness(),
                RandomContrast(),
                ConvertColor(transform='HSV'),
                RandomSaturation(),
                RandomHue(),
                ConvertColor(current='HSV', transform='BGR'),
            ]),
            alb.Compose([
                RandomBrightness(),
                ConvertColor(transform='HSV'),
                RandomSaturation(),
                RandomHue(),
                ConvertColor(current='HSV', transform='BGR'),
                RandomContrast()
            ])
        ]

    def apply(self, img: np.ndarray, **params):
        im = img.copy().astype(np.float32)
        if np.random.randint(2):
            res = self.pd[0](image=im)
        else:
            res = self.pd[1](image=im)
        return res.get('image').astype(np.uint8)
        # return self.rand_light_noise(im, boxes, labels)

    def apply_to_bboxes(self, bboxes, **params):
        return bboxes


class Expand(alb.DualTransform):
    def __init__(
            self,
            min_rate: float = 0.35,
            max_rate: float = 0.85,
            always_apply: bool = False,
            p: float = 0.5
    ):
        super().__init__(
            always_apply,
            p
        )
        self.min_rate = min_rate  # min old_img_h/new_img_h
        self.max_rate = max_rate  # max old_img_h/new_img_h
        assert self.max_rate >= self.min_rate
        self.back_ground_rate: float = None  # old_img_h/new_img_h

        self.shift_rate_right: float = None
        self.shift_right_pix: int = None  # old image shift right pix(on new image)

        self.shift_rate_down: float = None
        self.shift_down_pix: int = None  # old image shift down pix(on new image)

        self.old_shape: Tuple[int, int, int] = None
        self.new_shape: Tuple[int, int, int] = None

    def apply(self, img: np.ndarray, **params):
        self.back_ground_rate = np.random.uniform(self.min_rate, self.max_rate)
        self.shift_rate_right = np.random.random()
        self.shift_rate_down = np.random.random()

        h, w, c = img.shape
        self.old_shape = img.shape

        new_h, new_w = int(h / self.back_ground_rate), int(w / self.back_ground_rate)
        new_img = np.zeros(shape=(new_h, new_w, c), dtype=np.uint8)
        self.new_shape = (new_h, new_w, c)

        self.shift_right_pix = int((new_w - w) * self.shift_rate_right)

        self.shift_down_pix = int((new_h - h) * self.shift_rate_down)

        new_img[self.shift_down_pix:self.shift_down_pix + h,
        self.shift_right_pix: self.shift_right_pix + w, :] = img

        return new_img

    def apply_to_bboxes(self, bboxes, **params):
        h, w, c = self.old_shape
        new_h, new_w = self.new_shape[0], self.new_shape[1]
        new_bboxes = [
            (
                (box[0] * w + self.shift_right_pix) / new_w,
                (box[1] * h + self.shift_down_pix) / new_h,
                (box[2] * w + self.shift_right_pix) / new_w,
                (box[3] * h + self.shift_down_pix) / new_h,
                box[-1]
            ) for box in bboxes
        ]
        return new_bboxes


class VocDataSet(Dataset):
    def __init__(
            self,
            root: str,
            years: list,
            train: bool,
            image_size: Union[int, tuple],
            mean: list = [0.485, 0.456, 0.406],
            std: list = [0.229, 0.224, 0.225],
            transform: alb.Compose = None,
    ):
        super().__init__()
        self.root = root
        self.years = years
        self.train = train

        if self.train:
            self.data_type = 'trainval'
        else:
            self.data_type = 'test'
            self.years = ['2007']

        self.image_size = image_size if isinstance(image_size, int) else image_size[0]

        if transform is None:
            self.transform = alb.Compose([
                alb.Resize(image_size, image_size),
                alb.Normalize(mean, std)
            ], bbox_params=alb.BboxParams(format='pascal_voc'))
        else:
            self.transform = transform

        self.images_objects_masks_path = self.__get_all_path()

    def __get_all_path(
            self
    ) -> List:
        res = []
        if self.train:
            for year in self.years:
                root_path = os.path.join(
                    self.root,
                    year,
                    self.data_type,
                )
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
                txt_file_name = os.path.join(
                    root_path,
                    'ImageSets',
                    'Main',
                    '{}.txt'.format(self.data_type)
                )
                with open(txt_file_name, 'r') as f:
                    temp = f.readlines()
                    xml_file_names = [val[:-1] + '.xml' for val in temp]

                res += [
                    (
                        os.path.join(images_path, xml_file_name[:-3] + 'jpg'),
                        os.path.join(objects_path, xml_file_name),
                    ) for xml_file_name in xml_file_names
                ]
        else:
            for year in self.years:
                root_path = os.path.join(
                    self.root,
                    year,
                    self.data_type,
                )
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
                anno_path = os.path.join(
                    root_path,
                    'Annotations'
                )
                xml_file_names = os.listdir(anno_path)
                res += [
                    (
                        os.path.join(images_path, xml_file_name[:-3] + 'jpg'),
                        os.path.join(objects_path, xml_file_name),
                    ) for xml_file_name in xml_file_names
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

    def pull_an_image(
            self,
            index: int
    ) -> List:
        res = []
        image_path, obj_path = self.images_objects_masks_path[index]

        image = CV2.imread(image_path)
        res.append(image)

        objects_vec = self.read_xml_objects(obj_path)
        res.append(objects_vec)

        return res

    def __len__(self):
        return len(self.images_objects_masks_path)

    def __getitem__(
            self,
            index
    ):
        image, objects_vec = self.pull_an_image(index)

        image = CV2.cvtColorToRGB(image)

        res = self.transform(image=image, bboxes=objects_vec)

        new_image = res.get('image')
        new_obj_vec = res.get('bboxes')

        new_obj_vec = [[obj[-1], *(obj[:-1])] for obj in new_obj_vec]

        return torch.tensor(new_image, dtype=torch.float32).permute(2, 0, 1), new_obj_vec

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        imgs = batch[0]
        objects = batch[1]

        del batch
        return torch.stack(imgs), objects


def get_voc_data_loader(
        root: str,
        years: List[str],
        train: bool,
        image_size: int,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
        trans_form: alb.Compose = None,
        batch_size: int = 8,
        num_workers: int = 4,
):
    data_set = VocDataSet(
        root,
        years,
        train,
        image_size,
        mean,
        std,
        trans_form
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


def debug_pull_an_image():
    d = VocDataSet(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        416,
    )

    img, obj_vec = d.pull_an_image(index=np.random.randint(0, len(d)))
    debug_for_show(img, obj_vec)


def debug_transform():
    trans = alb.Compose([
        # alb.GaussNoise(),
        # alb.HueSaturationValue(),
        # alb.RandomBrightnessContrast(),
        # alb.ColorJitter(),
        # alb.Blur(3),
        # alb.HorizontalFlip(),
        # alb.Rotate(limit=(-30, 30), p=1.0),
        # alb.RandomResizedCrop(
        #     416,
        #     416,
        #     p=1.0,
        #     scale=(0.75, 1.0)
        # ),
        Expand(0.25, 1.0, p=1),
        # alb.GaussNoise(var_limit=(60, 150), p=1),

        alb.Resize(416, 416),

    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    d = VocDataSet(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        416,
        transform=trans,
    )

    img, obj_vec = d.pull_an_image(index=np.random.randint(0, len(d)))
    debug_for_show(img.copy(), obj_vec, pre_fix='old --> ')
    res = trans(image=img, bboxes=obj_vec)
    new_image = res.get('image')
    new_obj_vec = res.get('bboxes')
    debug_for_show(new_image.copy(), new_obj_vec, pre_fix='new --> ')


def debug_dataset():
    trans = alb.Compose([
        alb.Rotate(),
        alb.RandomBrightnessContrast(),
        alb.HorizontalFlip(),
        alb.HueSaturationValue(),
        alb.Resize(416, 416),
        alb.Normalize(
            mean=[0.406, 0.456, 0.485],
            std=[0.225, 0.224, 0.229]
        )
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    data_set = VocDataSet(
        '/home/dell/data/DataSet/VOC/',
        ['2007', '2012'],
        True,
        416,
        transform=trans,
    )
    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        data_set,
        shuffle=True,
        batch_size=8,
        collate_fn=data_set.collate_fn
    )

    for _, (images, objects) in enumerate(data_loader):
        print(images)
        print(objects)


if __name__ == '__main__':
    # debug_pull_an_image()
    debug_transform()
    # debug_dataset()
