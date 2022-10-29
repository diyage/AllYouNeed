"""
This packet is used for visualization.
"""
from abc import abstractmethod
from typing import Union
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from .predictor import DevPredictor
from .model import DevModel
from .tools import DevTool
from Package.BaseDev import BaseVisualizer, CV2


class DevVisualizer(BaseVisualizer):
    def __init__(
            self,
            model: DevModel,
            predictor: DevPredictor,
            class_colors: list,
            iou_th_for_make_target: float
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.predictor = predictor

        self.pre_anchor_w_h_rate = self.predictor.pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = self.predictor.image_shrink_rate
        self.grid_number = None

        self.image_size = None
        self.change_image_wh(self.predictor.image_size)

        self.kinds_name = self.predictor.kinds_name
        self.iou_th_for_show = self.predictor.iou_th

        self.class_colors = class_colors
        self.iou_th_for_make_target = iou_th_for_make_target

    @staticmethod
    def visualize(
            img: Union[torch.Tensor, np.ndarray],
            predict_name_pos_score: list,
            saved_path: str,
            class_colors: list,
            kinds_name: list,
    ):
        """
        Args:
            img: just one image
            predict_name_pos_score: [kps0, kps1, ...]
            saved_path:
            class_colors: [color0, color1, ...]
            kinds_name: [kind_name0, kind_name1, ...]

        Returns:
        """
        assert len(img.shape) == 3

        if not isinstance(img, np.ndarray):
            img = DevTool.image_tensor_to_np(img)

        for box in predict_name_pos_score:
            if len(box) != 3:
                continue
            predict_kind_name, abs_double_pos, prob_score = box
            color = class_colors[kinds_name.index(predict_kind_name)]

            CV2.rectangle(img,
                          start_point=(int(abs_double_pos[0]), int(abs_double_pos[1])),
                          end_point=(int(abs_double_pos[2]), int(abs_double_pos[3])),
                          color=color,
                          thickness=2)

            # CV2.rectangle(img,
            #               start_point=(int(abs_double_pos[0]), int(abs_double_pos[1] - 20)),
            #               end_point=(int(abs_double_pos[2]), int(abs_double_pos[1])),
            #               color=color,
            #               thickness=-1)

            scale = 0.5
            CV2.putText(img,
                        '{}:{:.2%}'.format(predict_kind_name, prob_score),
                        org=(int(abs_double_pos[0]), int(abs_double_pos[1] - 5)),
                        font_scale=scale,
                        # color=tuple([255 - val for val in color]),
                        color=(0, 0, 0),
                        back_ground_color=color
                        )

        CV2.imwrite(saved_path, img)

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ):
        pass

    def detect_one_image(
            self,
            image: Union[torch.Tensor, np.ndarray],
            saved_path: str,
    ):
        if isinstance(image, np.ndarray):
            image = CV2.resize(image, new_size=(416, 416))
            print('We resize the image to (416, 416), that may not be what you want!' +
                  'please resize your image before using this method!')
            image = DevTool.image_np_to_tensor(image)

        out_dict = self.model(image.unsqueeze(0).to(self.device))
        pre_kps_s = self.predictor.decode_one_predict(
            out_dict
        )
        self.visualize(
            image,
            pre_kps_s,
            saved_path=''.format(saved_path),
            class_colors=self.class_colors,
            kinds_name=self.kinds_name
        )

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):
        os.makedirs(saved_dir, exist_ok=True)
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            if batch_id == 10:
                break

            self.model.eval()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            output = self.model(images)

            gt_decode = self.predictor.decode_target(targets)
            pre_decode = self.predictor.decode_predict(output)

            for image_index in range(images.shape[0]):
                self.visualize(
                    images[image_index],
                    gt_decode[image_index],
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

                self.visualize(
                    images[image_index],
                    pre_decode[image_index],
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

