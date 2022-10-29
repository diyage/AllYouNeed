
'''
Define dataset/dataloader
Function --get_voc_data_loader-- will be what you want!
'''
import os.path
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.transforms as transforms
import numpy as np
from Package.BaseDev import CV2
from typing import List, Union

'''
Used for data augmentation.
SSDAugmentation --> for train
BaseAugmentation --> for test/eval
'''

from Package.BaseDev import BaseTool
import torch
import numpy as np
import types
import random
# from numpy import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class MyCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):

        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes=None, labels=None):
        image = BaseTool.image_np_to_tensor(image, self.mean, self.std)
        return image, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = CV2.cvtColor(image, CV2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = CV2.cvtColor(image, CV2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        # self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if np.random.randint(2):
            distort = MyCompose(self.pd[:-1])
        else:
            distort = MyCompose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels
        # return self.rand_light_noise(im, boxes, labels)


class Resize(object):
    def __init__(self, size=416):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        old_h, old_w = image.shape[0], image.shape[1]
        image = CV2.resize(image, (self.size, self.size))

        boxes[:, [0, 2]] /= (old_w/self.size)
        boxes[:, [1, 3]] /= (old_h/self.size)
        boxes = np.clip(boxes, a_min=0, a_max=self.size-1)

        return image, boxes, labels


class SSDAugmentation(object):
    def __init__(
            self,
            size: int = 416,
            mean=(0.406, 0.456, 0.485),
            std=(0.225, 0.224, 0.229),
            augment: MyCompose = None
    ):
        self.mean = mean
        self.std = std
        self.size = size
        if augment is None:
            self.augment = MyCompose([
                ConvertFromInts(),
                PhotometricDistort(),  # epoch_50 67.82% epoch_90 69.57%
                Expand(self.mean),  # epoch_50 66.50% epoch_90 67.54%
                RandomSampleCrop(),  # epoch_50 63.31% epoch_90 63.30%
                RandomMirror(),
                Resize(self.size),
                Normalize(self.mean, self.std)
            ])
        else:
            self.augment = augment

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class BaseAugmentation(object):
    def __init__(self,
                 size: int = 416,
                 mean=(0.406, 0.456, 0.485),
                 std=(0.225, 0.224, 0.229)
                 ):
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = MyCompose([
            ConvertFromInts(),
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

###################################################################################


class XMLTranslate:
    def __init__(self, root_path: str, file_name: str):
        if not root_path.endswith('/'):
            root_path += '/'

        self.annotations_path = root_path + 'Annotations/'
        self.images_path = root_path + 'JPEGImages/'

        self.root = ET.parse(self.annotations_path + file_name).getroot()  # type: xml.etree.ElementTree.Element
        self.img = None   # type:np.ndarray
        self.img_file_name = None  # type:str
        self.img_size = None  # type:tuple
        self.objects = []  # type:list
        self.__set_info()

    def __set_info(self):

        self.img_file_name = self.root.find('filename').text.strip()
        self.img = CV2.imread(self.images_path + self.img_file_name)

        for size in self.root.iter('size'):
            img_w = float(size.find('width').text.strip())
            img_h = float(size.find('height').text.strip())
            img_c = int(size.find('depth').text.strip())
            self.img_size = (img_w, img_h, img_c)

        for obj in self.root.iter('object'):
            kind = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            a = float(bbox.find('xmin').text.strip())  # point to x_axis dist
            b = float(bbox.find('ymin').text.strip())  # point to y_axis dist
            m = float(bbox.find('xmax').text.strip())
            n = float(bbox.find('ymax').text.strip())
            self.objects.append((kind, a, b, m, n))

    def resize(self, new_size: tuple = (448, 448)):
        self.img = CV2.resize(self.img, new_size)
        old_w = self.img_size[0]
        old_h = self.img_size[1]
        new_w = new_size[0]
        new_h = new_size[1]
        self.img_size = (new_w, new_h, self.img_size[2])

        for i in range(len(self.objects)):
            new_object = (
                self.objects[i][0],
                self.objects[i][1] / (old_w / new_w),
                self.objects[i][2] / (old_h / new_h),
                self.objects[i][3] / (old_w / new_w),
                self.objects[i][4] / (old_h / new_h)
            )
            self.objects[i] = new_object

    def get_image_size(self) -> tuple:
        return self.img_size

    def get_image_name(self) -> str:
        return self.img_file_name

    def get_objects(self) -> list:
        return self.objects

    def print(self):
        print("image name: {}".format(self.img_file_name))
        print("image size: {}".format(self.img_size))
        print("objects:")
        for val in self.objects:
            print("kind: {}, box: ({},{},{},{})".format(val[0], val[1], val[2], val[3], val[4]))


class VOCDataSet(Dataset):
    def __init__(
            self,
            root: str,
            years: list,
            train: bool = True,
            image_size: tuple = (448, 448),
            transform: Union[SSDAugmentation, BaseAugmentation] = None,
    ):
        # .../VOC/year/trainval(or test)/ ----
        super().__init__()
        self.root = root
        self.years = years
        self.train = train
        if self.train:
            self.data_type = 'trainval'
        else:
            self.data_type = 'test'
        self.image_size = image_size

        if transform is None:
            self.transform = BaseAugmentation()
        else:
            self.transform = transform

        self.image_and_xml_path_info = self.__get_image_and_xml_file_abs_path()

    def __get_image_and_xml_file_abs_path(self) -> list:
        res = []
        if self.train:
            for year in self.years:
                root_path = os.path.join(
                    self.root,
                    year,
                    self.data_type,
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

                res += [(root_path, xml_file_name) for xml_file_name in xml_file_names]
        else:
            for year in self.years:
                root_path = os.path.join(
                    self.root,
                    year,
                    self.data_type,
                )
                anno_path = os.path.join(
                    root_path,
                    'Annotations'
                )
                xml_file_names = os.listdir(anno_path)
                res += [(root_path, xml_file_name) for xml_file_name in xml_file_names]
        return res

    def __len__(self):
        return len(self.image_and_xml_path_info)

    def __get_image_label(
            self,
            index,
    ) -> tuple:
        root_path, xml_file_name = self.image_and_xml_path_info[index]
        xml_trans = XMLTranslate(root_path=root_path, file_name=xml_file_name)
        # xml_trans.resize(new_size=self.image_size)

        img, label = xml_trans.img, xml_trans.objects
        boxes = []
        classes = []

        for val in label:
            classes.append(val[0])
            boxes.append(val[1: 5])

        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes)

        return img, boxes, classes

    def __getitem__(self, index):
        img, boxes, classes = self.__get_image_label(index)

        new_img_tensor, new_boxes, new_classes = self.transform(
            img,
            boxes,
            classes
        )
        new_label = []
        for i in range(new_classes.shape[0]):
            new_label.append(
                (new_classes[i], *new_boxes[i].tolist())
            )

        return new_img_tensor, new_label

    @staticmethod
    def collate_fn(batch):
        # batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
        batch = list(zip(*batch))
        imgs = batch[0]
        labels = batch[1]
        del batch
        return torch.stack(imgs), labels


#######################################################


def get_voc_data_loader(
        root_path: str,
        years: list,
        image_size: tuple,
        batch_size: int,
        train: bool = True,
        num_workers: int = 0,
        mean: List[float] = [0.5, 0.5, 0.5],
        std: List[float] = [0.5, 0.5, 0.5],
):

    if train:
        transform_train = SSDAugmentation(
            size=image_size[0],
            mean=mean,
            std=std
        )

        train_d = VOCDataSet(
            root=root_path,
            years=years,
            train=True,
            image_size=image_size,
            transform=transform_train
        )

        train_l = DataLoader(train_d,
                             batch_size=batch_size,
                             collate_fn=VOCDataSet.collate_fn,
                             shuffle=True,
                             num_workers=num_workers)
        return train_l
    else:
        transform_test = BaseAugmentation(
            size=image_size[0],
            mean=mean,
            std=std
        )

        test_d = VOCDataSet(
            root=root_path,
            years=years,
            train=False,
            image_size=image_size,
            transform=transform_test
        )

        test_l = DataLoader(test_d,
                            batch_size=batch_size,
                            collate_fn=VOCDataSet.collate_fn,
                            shuffle=False,
                            num_workers=num_workers)
        return test_l


if __name__ == '__main__':
    # import numpy as np
    img = CV2.imread(r'/home/dell/data/DataSet/VOC/2007/test/JPEGImages/000001.jpg')
    l = [('dog', 20, 25, 120, 260)]
    boxes = []
    classes = []
    for val in l:
        classes.append(val[0])
        boxes.append(val[1: 5])

    boxes = np.array(boxes, dtype=np.float32)
    classes = np.array(classes)

    CV2.imshow('old', np.array(img, dtype=np.uint8))
    CV2.waitKey(0)

    R = SSDAugmentation(
        augment=MyCompose([
            # ConvertFromInts(),
            # PhotometricDistort(),
            # Expand(mean=(0.5, 0.5, 0.5)),
            RandomSampleCrop(),
            # RandomMirror(),
            Resize(416),
        ])
    )

    new_img, new_boxes, new_classes = R(img, boxes, classes)
    new_l = []
    for i in range(new_classes.shape[0]):
        new_l.append((new_classes[i], *new_boxes[i].tolist()))
    a = ['pig', 'cat', 'dog']

    CV2.imshow('new', np.array(new_img, dtype=np.uint8))
    CV2.waitKey(0)