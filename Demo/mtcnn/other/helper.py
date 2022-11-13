from .config import MTCNNConfig
from .MTCNN_DataSet import MTCNNDataSet
from Package.DataSet.FacialDetectionAndKeyPoints.WiderFace import WiderFace
from Package.DataSet.FacialDetectionAndKeyPoints.CelebA import CelebA
from Package.Task.FacialDetectionAndKeyPoints.MTCNN import *
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
import json
import cv2
import numpy as np
import random
import albumentations as alb


def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter, (box_area + area - inter))
    # ovr = inter / (box_area + area - inter)
    return ovr


class MTCNNHelper:
    def __init__(
            self,
            model: MTCNNModel,
            config: MTCNNConfig,
            restore_epoch: int = -1,
            restore_prefix: str = '',
    ):
        self.model = model
        self.device = config.train_config.device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch, restore_prefix)

        self.trainer = MTCNNTrainer(
            self.model,
        )

        self.visualizer = MTCNNVisualizer(

        )
        self.predictor = MTCNNPredictor(
            model,
            mean=self.config.data_config.mean,
            std=self.config.data_config.std
        )

    def restore(
            self,
            epoch: int,
            prefix: str = ''
    ):
        self.restore_epoch = epoch
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth/'
        saved_file_name = '{}/{}_{}.pth'.format(saved_dir, prefix, epoch)
        saved_state_dict = torch.load(saved_file_name)
        for net_type in ['p', 'r', 'o']:
            self.model.net_map[net_type].load_state_dict(saved_state_dict['{}_state_dict'.format(net_type)])

    def save(
            self,
            epoch: int,
            prefix: str = ''
    ):
        # save model
        self.model.eval()
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth/'
        os.makedirs(saved_dir, exist_ok=True)

        saved_state_dict = {
            '{}_state_dict'.format(net_type): self.model.net_map[net_type].state_dict() for net_type in ['p', 'r', 'o']
        }
        torch.save(saved_state_dict, '{}/{}_{}.pth'.format(saved_dir, prefix, epoch))

    def generate_images_for_key_point(
            self,
            used_for_net_type: str,
            output_folder: str,
    ):
        def split():
            with open('{}/info.json'.format(output_folder), mode='r') as f:
                g_info: dict = json.load(f)
            all_abs_path = list(g_info['key_point'].keys())
            all_abs_path = np.random.choice(all_abs_path, size=(len(all_abs_path),), replace=False).tolist()
            cut = int(0.8 * len(all_abs_path))
            train_json = all_abs_path[:cut]
            test_json = all_abs_path[cut:]
            with open('{}/train.json'.format(output_folder), mode='w') as f:
                json.dump(train_json, f)
            with open('{}/test.json'.format(output_folder), mode='w') as f:
                json.dump(test_json, f)

        """
                For training P-net/R-Net/O-Net, crop 12*12 images from original images.
                The Generated file will be saved in "output_folder"

                output_folder------image
                                |--info.json
                                |--train.json
                                |--test.json

                :param used_for_net_type:
                :param output_folder: saved cropped images
                :return:
        """
        assert used_for_net_type in ['p', 'r', 'o']

        """
        collect train(0)/val(1) from CelebA (using list_eval_partition.txt)
        """
        original_info = CelebA.get_all_original_info(self.config.data_config.CelebA_root)

        """
        crop all original images and save them to positive_path/negative_path/part_path/info.json(gt_box)
        """
        if os.path.exists('{}/info.json'.format(output_folder)):
            print(
                '\nAlready generate images for training {}-Net on CelebA(key_point)! So we do not process again!'.format(
                    used_for_net_type
                ))
            print("\nWe will re-split train/val.")
            split()

            return None

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder, exist_ok=True)

        new_image_path = os.path.join(output_folder, 'image')
        os.makedirs(new_image_path, exist_ok=True)

        crop_size = self.config.data_config.images_cropped_size[used_for_net_type][0]
        original_image_abs_path_vec = list(original_info.get('position').keys())
        total_num = 0
        generate_info = {
            'key_point': {}
        }
        for original_image_path in tqdm(
                original_image_abs_path_vec,
                desc='Generate cropped images for training {}-Net key-point using CelebA.'.format(used_for_net_type),
                position=0
        ):
            bbox = original_info.get('position')[original_image_path]
            landmark = original_info.get('key_point')[original_image_path]
            img = cv2.imread(original_image_path)
            img_w = img.shape[0]
            img_h = img.shape[1]

            left = bbox[0]
            top = bbox[1]
            w = bbox[2]
            h = bbox[3]

            if w <= 0 or h <= 0:
                continue

            right = bbox[0] + w + 1
            bottom = bbox[1] + h + 1

            # Crop the face image.
            face_img = img[top: bottom, left: right]

            # Resize the image
            face_img = cv2.resize(face_img, (crop_size, crop_size))

            # Resize landmark as (5, 2)
            landmark = np.array(landmark)
            landmark.resize(5, 2)

            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            landmark_gtx = (landmark[:, 0] - left) / w
            landmark_gty = (landmark[:, 1] - top) / h
            landmark_gt = np.concatenate([landmark_gtx, landmark_gty]).tolist()

            total_num += 1
            resize_img_abs_path = os.path.join(
                new_image_path,
                str(total_num) + '.jpg'
            )
            cv2.imwrite(resize_img_abs_path, face_img)
            generate_info['key_point'][resize_img_abs_path] = landmark_gt

            if max(w, h) < 40 or left < 0 or right <= 0 or min(w, h) <= 0:
                continue

            # random shift
            for i in range(5):
                bbox_size = np.random.randint(
                    int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                low = int(-w * 0.2)
                high = int(w * 0.2)

                if low >= high:
                    continue
                delta_x = np.random.randint(low, high)
                delta_y = np.random.randint(low, high)
                nx1 = int(max(left + w / 2 - bbox_size / 2 + delta_x, 0))
                ny1 = int(max(top + h / 2 - bbox_size / 2 + delta_y, 0))

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])
                gt_box = np.array([left, top, right, bottom])

                iou = IoU(crop_box, np.expand_dims(gt_box, 0))

                if iou > 0.65:
                    landmark_croppedx = (landmark[:, 0] - nx1) / bbox_size
                    landmark_croppedy = (landmark[:, 1] - ny1) / bbox_size
                    landmark_gt = np.concatenate(
                        [landmark_croppedx, landmark_croppedy]).tolist()
                    cropped_img = img[ny1: ny2, nx1: nx2]
                    cropped_img = cv2.resize(cropped_img, (crop_size, crop_size))

                    total_num += 1
                    resize_img_abs_path = os.path.join(
                        new_image_path,
                        str(total_num) + '.jpg'
                    )
                    cv2.imwrite(resize_img_abs_path, cropped_img)
                    generate_info['key_point'][resize_img_abs_path] = landmark_gt

        with open('{}/info.json'.format(output_folder), mode='w') as f:
            json.dump(generate_info, f)

        print('\nSplit train/val')
        split()

    def generate_images_for_detection(
            self,
            used_for_net_type: str,
            output_folder: str,
    ):
        def split():
            train_json = {}
            test_json = {}
            for data_type in ['positive', 'negative', 'part']:
                data_path = os.path.join(
                    output_folder,
                    data_type
                )
                img_name_vec = os.listdir(data_path)
                img_name_vec = [os.path.join(data_path, img_name) for img_name in img_name_vec]
                img_name_vec = np.random.choice(img_name_vec, size=(len(img_name_vec),), replace=False).tolist()
                cut = int(0.8 * len(img_name_vec))
                train_json[data_type] = img_name_vec[:cut]
                test_json[data_type] = img_name_vec[cut:]

            with open('{}/train.json'.format(output_folder), mode='w') as f:
                json.dump(train_json, f)
            with open('{}/test.json'.format(output_folder), mode='w') as f:
                json.dump(test_json, f)

        """
        For training P-net/R-Net/O-Net, crop positive(0), negative(1) and part(2) from original images.
        The Generated file will be saved in "output_folder"

        output_folder------positive
                        |--negative
                        |--part
                        |--info.json
                        |--train.json
                        |--test.json

        """
        assert used_for_net_type in ['p', 'r', 'o']

        """
        collect train/val/test from WiderFace
        """
        original_info = WiderFace.get_all_original_info(self.config.data_config.WiderFace_root)

        """
        crop all original images and save them to positive_path/negative_path/part_path/info.json(gt_box)
        """
        if os.path.exists('{}/info.json'.format(output_folder)):
            print(
                '\nAlready generate images for training {}-Net on WiderFace(position)! So we do not process again!'.format(
                    used_for_net_type
                ))
            print("\nWe will re-split train/val.")
            split()

            return None

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder, exist_ok=True)

        positive_path = os.path.join(output_folder, 'positive')
        os.makedirs(positive_path, exist_ok=True)

        negative_path = os.path.join(output_folder, 'negative')
        os.makedirs(negative_path, exist_ok=True)

        part_path = os.path.join(output_folder, 'part')
        os.makedirs(part_path, exist_ok=True)

        crop_size = self.config.data_config.images_cropped_size[used_for_net_type][0]

        if used_for_net_type == 'p':
            generate_info = {
                'position': {}
            }
            total_pos_num = 0
            total_neg_num = 0
            total_part_num = 0
            for original_image_path, original_position_vec in tqdm(
                    original_info['position'].items(),
                    desc='generate cropped images for P-Net',
                    position=0
            ):
                img = cv2.imread(original_image_path)
                boxes = np.array(original_position_vec, np.int32)
                height, width, _ = img.shape

                neg_num = 0
                pos_num = 0
                part_num = 0
                """
                Record the number of positive, negative and part examples.
                """
                while neg_num < 50:

                    size = np.random.randint(crop_size, min(width, height) / 2)

                    nx = np.random.randint(0, width - size)
                    ny = np.random.randint(0, height - size)

                    crop_box = np.array([nx, ny, nx + size, ny + size])
                    iou = IoU(crop_box, boxes)

                    if np.max(iou) < 0.3:
                        # Iou with all gts must below 0.3
                        cropped_img = img[ny: ny + size, nx: nx + size, :]
                        resize_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                        total_neg_num += 1
                        neg_num += 1

                        resize_img_abs_path = os.path.join(
                            negative_path,
                            str(total_neg_num) + '.jpg'
                        )
                        generate_info['position'][resize_img_abs_path] = []
                        cv2.imwrite(resize_img_abs_path, resize_img)

                for box in boxes:
                    # box (x_left, y_top, x_right, y_bottom)
                    x1, y1, x2, y2 = box
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1

                    # ignore small faces
                    # in case the ground truth boxes of small faces are not accurate
                    if max(w, h) <= 40 or x1 <= 0 or y1 <= 0 or w <= 0 or h <= 0:
                        continue

                    # generate negative examples that have overlap with gt
                    for i in range(5):
                        size = np.random.randint(crop_size, min(width, height) / 2)
                        # delta_x and delta_y are offsets of (x1, y1)
                        delta_x = np.random.randint(max(-size, -x1), w)
                        delta_y = np.random.randint(max(-size, -y1), h)

                        nx1 = max(0, x1 + delta_x)
                        ny1 = max(0, y1 + delta_y)

                        if nx1 + size > width or ny1 + size > height:
                            continue
                        crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                        iou = IoU(crop_box, boxes)

                        cropped_img = img[ny1: ny1 + size, nx1: nx1 + size, :]
                        resize_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                        if np.max(iou) < 0.3:
                            # Iou with all gts must below 0.3
                            neg_num += 1
                            total_neg_num += 1
                            resize_img_abs_path = os.path.join(
                                negative_path,
                                str(total_neg_num) + '.jpg'
                            )
                            generate_info['position'][resize_img_abs_path] = []
                            cv2.imwrite(resize_img_abs_path, resize_img)

                    # generate positive examples and part faces
                    for i in range(20):
                        size = np.random.randint(int(min(w, h) * 0.8), 1 + np.ceil(1.25 * max(w, h)))

                        # delta here is the offset of box center
                        low = int(-w * 0.2)
                        high = int(w * 0.2)

                        if low >= high:
                            continue

                        delta_x = np.random.randint(low, high)
                        delta_y = np.random.randint(low, high)

                        nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                        ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                        nx2 = nx1 + size
                        ny2 = ny1 + size

                        if nx2 > width or ny2 > height:
                            continue
                        crop_box = np.array([nx1, ny1, nx2, ny2])

                        offset_x1 = (x1 - nx1) / float(size)
                        offset_y1 = (y1 - ny1) / float(size)
                        offset_x2 = (x2 - nx2) / float(size)
                        offset_y2 = (y2 - ny2) / float(size)

                        cropped_img = img[ny1: ny2, nx1: nx2, :]
                        resize_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                        box_ = box.reshape(1, -1)
                        if IoU(crop_box, box_) >= 0.65:
                            pos_num += 1
                            total_pos_num += 1
                            resize_img_abs_path = os.path.join(
                                positive_path,
                                str(total_pos_num) + '.jpg'
                            )
                            generate_info['position'][resize_img_abs_path] = [
                                offset_x1,
                                offset_y1,
                                offset_x2,
                                offset_y2
                            ]
                            cv2.imwrite(resize_img_abs_path, resize_img)

                        elif IoU(crop_box, box_) >= 0.4:
                            part_num += 1
                            total_part_num += 1
                            resize_img_abs_path = os.path.join(
                                part_path,
                                str(total_part_num) + '.jpg'
                            )
                            generate_info['position'][resize_img_abs_path] = [
                                offset_x1,
                                offset_y1,
                                offset_x2,
                                offset_y2
                            ]

                            cv2.imwrite(resize_img_abs_path, resize_img)

            with open('{}/info.json'.format(output_folder), mode='w') as f:
                json.dump(generate_info, f)

        else:
            """
            Generate cropped images for training R-Net or O-Net
            """
            generate_info = {
                'position': {}
            }
            total_pos_num = 0
            total_neg_num = 0
            total_part_num = 0
            self.model.eval()
            for original_image_path, original_position_vec in tqdm(
                    original_info['position'].items(),
                    desc='generate cropped images for {}-Net'.format('R' if used_for_net_type == 'r' else 'O'),
                    position=0
            ):
                img = cv2.imread(original_image_path)
                boxes = np.array(original_position_vec, dtype=np.int32)
                height, width, _ = img.shape

                neg_examples = []
                part_examples = []
                part_offsets = []

                neg_num = 0
                pos_num = 0
                part_num = 0
                """
                Record the number of positive, negative and part examples.
                """
                if used_for_net_type == 'r':
                    candidate_boxes: np.ndarray = self.predictor.detect(
                        img,
                        use_net_type='p',
                        stage_one_para={
                            'image_scale_rate': self.config.eval_config.image_scale_rate,
                            'min_size': self.config.data_config.images_cropped_size['p'][0],
                            'cls_threshold': self.config.train_config.stage_one_threshold['cls'],
                            'nms_threshold': self.config.train_config.stage_one_threshold['nms']
                        }
                    ).get('position')
                else:
                    candidate_boxes: np.ndarray = self.predictor.detect(
                        img,
                        use_net_type='pr',
                        stage_one_para={
                            'image_scale_rate': self.config.eval_config.image_scale_rate,
                            'min_size': self.config.data_config.images_cropped_size['p'][0],
                            'cls_threshold': self.config.train_config.stage_one_threshold['cls'],
                            'nms_threshold': self.config.train_config.stage_one_threshold['nms']
                        },
                        stage_two_para={
                            'cropped_size': self.config.data_config.images_cropped_size['r'][0],
                            'batch_size': self.config.train_config.batch_size,
                            'cls_threshold': self.config.train_config.stage_two_threshold['cls'],
                            'nms_threshold': self.config.train_config.stage_two_threshold['nms']
                        }

                    ).get('position')
                candidate_boxes = candidate_boxes.astype(np.int32)
                for c_box in candidate_boxes:
                    nx1 = c_box[0]
                    ny1 = c_box[1]
                    nx2 = c_box[2]
                    ny2 = c_box[3]

                    w = nx2 - nx1 + 1
                    h = ny2 - ny1 + 1

                    if nx2 > width or ny2 > height or nx1 <= 0 or ny1 <= 0 or w <= 0 or h <= 0:
                        continue

                    cropped_img = img[c_box[1]: c_box[3], c_box[0]: c_box[2], :]
                    resize_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

                    iou = IoU(c_box, boxes)
                    max_iou = iou.max()

                    if max_iou < 0.3:
                        neg_num += 1
                        neg_examples.append(resize_img)
                        continue

                    max_index = iou.argmax()

                    x1, y1, x2, y2 = boxes[max_index]

                    offset_x1 = (x1 - nx1) / float(w)
                    offset_y1 = (y1 - ny1) / float(h)
                    offset_x2 = (x2 - nx2) / float(w)
                    offset_y2 = (y2 - ny2) / float(h)

                    if max_iou >= 0.65:
                        pos_num += 1
                        total_pos_num += 1

                        resize_img_abs_path = os.path.join(
                            positive_path,
                            str(total_pos_num) + '.jpg'
                        )
                        generate_info['position'][resize_img_abs_path] = [
                            offset_x1,
                            offset_y1,
                            offset_x2,
                            offset_y2
                        ]

                        cv2.imwrite(resize_img_abs_path, resize_img)

                    elif max_iou >= 0.4:
                        part_num += 1
                        part_examples.append(resize_img)
                        part_offsets.append([offset_x1, offset_y1, offset_x2, offset_y2])

                # Prevent excessive negative samples
                if neg_num > 4 * pos_num:
                    neg_examples = random.sample(neg_examples, k=3 * pos_num)

                for i in neg_examples:
                    total_neg_num += 1
                    resize_img_abs_path = os.path.join(
                        negative_path,
                        str(total_neg_num) + '.jpg'
                    )

                    generate_info['position'][resize_img_abs_path] = []
                    cv2.imwrite(resize_img_abs_path, i)

                # Prevent excessive part samples
                if part_num > 2 * pos_num:
                    choiced_index = random.sample(list(range(part_num)), k=2 * pos_num)
                    part_examples = [part_examples[i] for i in choiced_index]
                    part_offsets = [part_offsets[i] for i in choiced_index]

                for i, offsets in zip(part_examples, part_offsets):
                    total_part_num += 1
                    resize_img_abs_path = os.path.join(
                        part_path,
                        str(total_part_num) + '.jpg'
                    )
                    offset_x1, offset_y1, offset_x2, offset_y2 = offsets
                    generate_info['position'][resize_img_abs_path] = [
                        offset_x1,
                        offset_y1,
                        offset_x2,
                        offset_y2
                    ]

                    cv2.imwrite(resize_img_abs_path, i)

            with open('{}/info.json'.format(output_folder), mode='w') as f:
                json.dump(generate_info, f)

        print('\nSplit train/val')
        split()

    def prepare_data_set_for_training_x_net(
            self,
            train_net_type: str
    ):
        assert train_net_type in ['p', 'r', 'o']
        """
                        prepare data_set
        """
        """
        prepare CelebA
        key-point 
        """

        print("\n>>> Prepare CelebA to {}/{}".format(
            self.config.ABS_PATH,
            self.config.data_config.CelebA_cache
        ))

        output_folder = os.path.join(
            self.config.ABS_PATH,
            self.config.data_config.CelebA_cache,
            'used_for_training_{}_net'.format(train_net_type)
        )
        self.generate_images_for_key_point(
            used_for_net_type=train_net_type,
            output_folder=output_folder
        )
        """
                prepare WiderFace to self.config.ABS_PATH/self.config.data_config.WiderFace_cache
                boxes
        """

        print("\n>>> Prepare WiderFace to {}/{}".format(
            self.config.ABS_PATH,
            self.config.data_config.WiderFace_cache
        ))

        output_folder = os.path.join(
            self.config.ABS_PATH,
            self.config.data_config.WiderFace_cache,
            'used_for_training_{}_net'.format(train_net_type)
        )
        self.generate_images_for_detection(
            used_for_net_type=train_net_type,
            output_folder=output_folder
        )

    def train_x_net(
            self,
            train_net_type='p'
    ):
        assert train_net_type in ['p', 'r', 'o']

        self.prepare_data_set_for_training_x_net(
            train_net_type=train_net_type
        )
        """
        Make data loader
        """
        cropped_celeb_a_root = os.path.join(
            self.config.ABS_PATH,
            self.config.data_config.CelebA_cache,
            'used_for_training_{}_net'.format(train_net_type)
        )
        cropped_wider_face_root = os.path.join(
            self.config.ABS_PATH,
            self.config.data_config.WiderFace_cache,
            'used_for_training_{}_net'.format(train_net_type)
        )
        trans = alb.Compose([
            alb.Resize(*self.config.data_config.images_cropped_size.get(train_net_type)),
            alb.Normalize(
                mean=self.config.data_config.mean,
                std=self.config.data_config.std
            )
        ])
        data_set = MTCNNDataSet(
            cropped_celeb_a_root,
            cropped_wider_face_root,
            train=True,
            transform=trans
        )
        data_loader = DataLoader(
            data_set,
            batch_size=self.config.train_config.batch_size,
            shuffle=True,
            num_workers=self.config.train_config.num_workers
        )
        loss_func = MTCNNLoss(
            self.config.train_config.cls_factor[train_net_type],
            self.config.train_config.box_factor[train_net_type],
            self.config.train_config.landmark_factor[train_net_type]
        )
        optimizer = torch.optim.Adam(
            self.model.net_map[train_net_type].parameters(),
            lr=self.config.train_config.lr[train_net_type]
        )

        for epoch in range(0, self.config.train_config.max_epoch_for_train[train_net_type]):

            loss_dict: dict = self.trainer.train_one_epoch(
                data_loader,
                loss_func,
                optimizer,
                now_epoch=epoch,
                max_epoch=self.config.train_config.max_epoch_for_train[train_net_type],
                train_net_type=train_net_type
            )

            print_info = '\n\nepoch: {} , loss info-->\n'.format(
                epoch
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if epoch % self.config.eval_config.eval_frequency == 0:
                # save model
                self.save(epoch, prefix='already_train_{}_'.format(train_net_type))

    def go(
            self,

    ):
        pass

