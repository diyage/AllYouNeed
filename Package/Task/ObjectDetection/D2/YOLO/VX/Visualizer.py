import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from Package.Task.ObjectDetection.D2.YOLO.VX.Typing import *
from Package.Task.ObjectDetection.D2.YOLO.VX.Model import YOLOVXModel
from Package.Task.ObjectDetection.D2.YOLO.VX.Predictor import YOLOVXPredictor
from Package.Task.ObjectDetection.D2.YOLO.VX.Tools import YOLOVXTool
from Package.BaseDev import BaseVisualizer, CV2


class YOLOVXVisualizer(BaseVisualizer):
    def __init__(
            self,
            model: YOLOVXModel,
            predictor: YOLOVXPredictor,
            class_colors: list,
            kinds_name: List[str],
            image_size: int = 640,
            image_shrink_rate: Tuple[int, int, int] = (8, 16, 32),
            multi_positives: bool = True
    ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.predictor = predictor

        self.image_size = image_size

        self.kinds_name = kinds_name
        self.iou_th_for_show = self.predictor.iou_th

        self.class_colors = class_colors
        self.image_shrink_rate = image_shrink_rate
        self.multi_positives = multi_positives

    @staticmethod
    def visualize(
            img: Union[torch.Tensor, np.ndarray],
            predict_kind_ind_pos_score: KPS_VEC,
            saved_path: str,
            class_colors: list,
            kinds_name: list,
    ):

        assert len(img.shape) == 3

        if not isinstance(img, np.ndarray):
            img = YOLOVXTool.image_tensor_to_np(img)

        for box in predict_kind_ind_pos_score:
            if len(box) != 3:
                continue
            predict_kind_ind, abs_double_pos, prob_score = box
            predict_kind_name = kinds_name[predict_kind_ind]
            color = class_colors[predict_kind_ind]
            CV2.rectangle(
                img,
                start_point=(int(abs_double_pos[0]), int(abs_double_pos[1])),
                end_point=(int(abs_double_pos[2]), int(abs_double_pos[3])),
                color=color,
                thickness=2
            )

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

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):
        os.makedirs(saved_dir, exist_ok=True)

        max_batch = len(data_loader_test)
        random_batch = np.random.choice(
            [ind for ind in range(max_batch)],
            size=(10, ),
            replace=False
        ).tolist()

        for batch_id, val in enumerate(tqdm(data_loader_test, desc=desc, position=0)):
            images: torch.Tensor = val[0]
            labels: List[LABEL] = val[1]

            if batch_id not in random_batch:
                continue

            self.model.eval()
            images = images.to(self.device)

            outputs = self.model(images)

            pre_decode: List[KPS_VEC] = self.predictor.post_process(outputs)

            for image_index in range(images.shape[0]):

                now_image_gt: KPS_VEC = [
                    (obj[0], obj[1], 1.0) for obj in labels[image_index]
                ]
                now_image_pre: KPS_VEC = pre_decode[image_index]

                self.visualize(
                    images[image_index],
                    now_image_gt,
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

                self.visualize(
                    images[image_index],
                    now_image_pre,
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

