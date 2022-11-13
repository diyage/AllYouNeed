from ..Dev import DevPredictor
from .Model import MTCNNModel
from .Tools import MTCNNTool
import typing
import numpy as np
import torch
import albumentations as alb


def no_grad(func):

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            ret = func(*args, **kwargs)
        return ret

    return wrapper


class MTCNNPredictor(DevPredictor):
    def __init__(
            self,
            model: MTCNNModel,
            mean: list,
            std: list,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.device
        self.mean = mean
        self.std = std

    def to_tensor(
            self,
            img: np.ndarray,
    ) -> torch.Tensor:
        trans = alb.Compose([
            alb.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
        if len(img.shape) == 3:
            x = trans(image=img).get('image')
            x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            return x
        else:
            vec = []
            for i in range(img.shape[0]):
                x = trans(image=img[i]).get('image')
                vec.append(x)

            vec = np.array(vec)
            x = torch.tensor(vec, dtype=torch.float32).permute(0, 3, 1, 2)
            return x

    @no_grad
    def stage_one(
            self,
            x: np.ndarray,
            image_scale_rate: float,
            min_size: int,
            cls_threshold: float,
            nms_threshold: float
    ) -> np.ndarray:
        """
        return candidate_box (after nms)
        """
        self.model.eval()
        res = MTCNNTool.get_more_scale_size_image(
            x,
            image_scale_rate,
            min_size
        )
        box_vec = np.empty(shape=(0, 4), dtype=np.float32)
        np_img_h, np_img_w, _ = x.shape

        for img, scale_h, scale_w in res:
            img_tensor = self.to_tensor(img)
            img_tensor = img_tensor.to(self.device)  # (1, 3, h, w)
            _, _, h, w = img_tensor.shape
            predict: dict = self.model.net_map['p'](img_tensor)
            _, _, m, n = predict.get('cls').shape

            split_predict = MTCNNTool.split_net_predict(predict, net_type='p')

            generated_candidate_box = MTCNNTool.generate_candidate_box(
                original_h=h,
                original_w=w,
                mapping_h=m,
                mapping_w=n,
                box_size=min_size
            )  # m * n * 4
            generated_candidate_box = generated_candidate_box.contiguous().view(-1, 4)  # (m*n, 4)
            generated_candidate_box = generated_candidate_box.to(img_tensor.device)

            """
            be careful, we do not need key_point for stage one
            """
            adjusted_box, _ = MTCNNTool.adjust_and_filter_candidate_box_and_landmark(
                generated_candidate_box,
                pre_cls=split_predict.get('cls'),
                pre_position_offset=split_predict.get('pos_offset'),
                pre_key_point_offset=split_predict.get('key_point'),
                score_threshold=cls_threshold,
                nms_threshold=nms_threshold
            )

            """
            scale on the original numpy image 
            """
            adjusted_box[:, [0, 2]] = adjusted_box[:, [0, 2]] / scale_w
            adjusted_box[:, [1, 3]] = adjusted_box[:, [1, 3]] / scale_h

            """
            filter out margin box
            """

            mask_w = (adjusted_box[:, 2] < np_img_w).float()
            mask_h = (adjusted_box[:, 3] < np_img_h).float()
            mask = mask_w * mask_h
            mask_bool = mask.bool()
            adjusted_box = adjusted_box[mask_bool]

            if adjusted_box.shape[0] != 0:
                box_vec = np.concatenate([box_vec, adjusted_box.cpu().detach().numpy()], axis=0)
        return box_vec

    @no_grad
    def stage_two(
            self,
            x: np.ndarray,
            candidate_box: np.ndarray,
            cropped_size: int,
            batch_size: int,
            cls_threshold: float,
            nms_threshold: float
    ) -> np.ndarray:
        """
            return adjust_candidate_box (after nms)
            candidate_box come from stage one

            x: (h, w, 3)

        """
        self.model.eval()

        if candidate_box.shape[0] == 0:
            return candidate_box

        box_vec = np.empty(shape=(0, 4), dtype=np.float32)
        np_img_h, np_img_w, _ = x.shape

        cropped_images, candidate_box = MTCNNTool.get_cropped_images_with_candidate_box(
            img=x,
            candidate_box=candidate_box,
            cropped_size=cropped_size
        )

        assert cropped_images.shape[0] == candidate_box.shape[0]

        num = cropped_images.shape[0]
        batch_ind = -1

        while True:
            batch_ind += 1
            i = batch_ind * batch_size

            if i >= num:
                break

            j = min((batch_ind + 1) * batch_size, num)

            img = cropped_images[i:j, ...]  # (batch, 24, 24, 3)
            img_tensor = self.to_tensor(img).to(self.device)  # (batch, 3, 24, 24)

            candidate_box_tensor = torch.tensor(
                candidate_box[i:j, ...],
                dtype=torch.float32
            ).to(self.device)  # (batch, 4)

            predict: dict = self.model.net_map['r'](img_tensor)
            split_predict = MTCNNTool.split_net_predict(predict, net_type='r')

            """
            be careful, we do not need key_point for stage two
            """

            adjusted_box, _ = MTCNNTool.adjust_and_filter_candidate_box_and_landmark(
                candidate_box_tensor,
                pre_cls=split_predict.get('cls'),
                pre_position_offset=split_predict.get('pos_offset'),
                pre_key_point_offset=split_predict.get('key_point'),
                score_threshold=cls_threshold,
                nms_threshold=nms_threshold
            )

            """
            filter out margin box
            """
            mask_w = (adjusted_box[:, 2] < np_img_w).float()
            mask_h = (adjusted_box[:, 3] < np_img_h).float()
            mask = mask_w * mask_h
            mask_bool = mask.bool()
            adjusted_box = adjusted_box[mask_bool]

            if adjusted_box.shape[0] != 0:
                box_vec = np.concatenate([box_vec, adjusted_box.cpu().detach().numpy()], axis=0)

        return box_vec

    @no_grad
    def stage_three(
            self,
            x: np.ndarray,
            adjust_candidate_box: np.ndarray,
            cropped_size: int,
            batch_size: int,
            cls_threshold: float,
            nms_threshold: float
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
            return adjust_2_candidate_box (after nms) and key-point
        """
        self.model.eval()

        candidate_box = adjust_candidate_box

        if candidate_box.shape[0] == 0:
            return candidate_box, np.empty(shape=(0, 10), dtype=np.float32)

        box_vec = np.empty(shape=(0, 4), dtype=np.float32)
        landmark_vec = np.empty(shape=(0, 10), dtype=np.float32)

        np_img_h, np_img_w, _ = x.shape

        cropped_images, candidate_box = MTCNNTool.get_cropped_images_with_candidate_box(
            img=x,
            candidate_box=candidate_box,
            cropped_size=cropped_size
        )

        assert cropped_images.shape[0] == candidate_box.shape[0]

        num = cropped_images.shape[0]
        batch_ind = -1

        while True:
            batch_ind += 1
            i = batch_ind * batch_size

            if i >= num:
                break

            j = min((batch_ind + 1) * batch_size, num)

            img = cropped_images[i:j, ...]  # (batch, 24, 24, 3)
            img_tensor = self.to_tensor(img).to(self.device)  # (batch, 3, 24, 24)

            candidate_box_tensor = torch.tensor(
                candidate_box[i:j, ...],
                dtype=torch.float32
            ).to(self.device)  # (batch, 4)

            predict: dict = self.model.net_map['o'](img_tensor)
            split_predict = MTCNNTool.split_net_predict(predict, net_type='o')

            """
            be careful, we need adjusted_box and key_point for stage three
            """

            adjusted_box, adjusted_landmark = MTCNNTool.adjust_and_filter_candidate_box_and_landmark(
                candidate_box_tensor,
                pre_cls=split_predict.get('cls'),
                pre_position_offset=split_predict.get('pos_offset'),
                pre_key_point_offset=split_predict.get('key_point'),
                score_threshold=cls_threshold,
                nms_threshold=nms_threshold
            )
            assert adjusted_box.shape[0] == adjusted_landmark.shape[0]
            """
            filter out margin box
            """
            mask_w = (adjusted_box[:, 2] < np_img_w).float()
            mask_h = (adjusted_box[:, 3] < np_img_h).float()
            mask = mask_w * mask_h
            mask_bool = mask.bool()

            adjusted_box = adjusted_box[mask_bool]
            adjusted_landmark = adjusted_landmark[mask_bool]

            if adjusted_box.shape[0] != 0:
                box_vec = np.concatenate([box_vec, adjusted_box.cpu().detach().numpy()], axis=0)
                landmark_vec = np.concatenate([landmark_vec, adjusted_landmark.cpu().detach().numpy()], axis=0)

        return box_vec, landmark_vec

    def detect(
            self,
            x: np.ndarray,
            use_net_type: str,
            stage_one_para: dict = {},
            stage_two_para: dict = {},
            stage_three_para: dict = {},
    ) -> typing.Dict:
        self.model.eval()

        assert use_net_type in ['p', 'pr', 'pro']
        """
        1)  If you want to generate cropped images for training R-Net, please
            set use_net_type as 'p', it will return the candidate_box.
        
        2)  If you want to generate cropped images for training O-Net, please
            set use_net_type as 'pr', it will return the adjust_candidate_box.
        
        3)  If you want to predict an image, please set use_net_type as 'pro'.
        """
        assert len(x.shape) == 3  # just one image
        """
        be careful, when you get predict of P-Net/R-Net/O-Net, how to get decode_box ?
        assume that you already get the offset from predict.(assume shape is (N, 4))
        then,
            tx1, ty1, tx2, ty2 = offset[0], offset[1], offset[2], offset[3]
            decode_box = candidate_box + offset * candidate_w_h_w_h
        Why?
            cause, situation of making ground-truth is:
                offset = (true_box - candidate_box_generated_by_random) / candidate_w_h_w_h
                
        So, if you want to decode true box from offset(get from predict), you need important candidate_box.
        
        But, where do/does candidate_box come from?
            We know, for offset of R-Net/O-Net, candidate_box come from last decoding process.
            For P-Net, candidate_box come from generating process(
                using grid map output from P-Net, 
                i.e, using mapping relationship between grid map and original image
                to generating lots of 12*12 candidate boxes
                ).
        """
        candidate_box = self.stage_one(
            x,
            image_scale_rate=stage_one_para.get('image_scale_rate'),
            min_size=stage_one_para.get('min_size'),
            cls_threshold=stage_one_para.get('cls_threshold'),
            nms_threshold=stage_one_para.get('nms_threshold')
        )

        if use_net_type == 'p':
            # print('>>> just use p-net detect')
            return {
                'position': candidate_box
            }

        adjust_candidate_box = self.stage_two(
            x,
            candidate_box=candidate_box,
            cropped_size=stage_two_para.get('cropped_size'),
            batch_size=stage_two_para.get('batch_size'),
            cls_threshold=stage_two_para.get('cls_threshold'),
            nms_threshold=stage_two_para.get('nms_threshold')
        )

        if use_net_type == 'pr':
            # print('>>> just use p-net and r-net detect')
            return {
                'position': adjust_candidate_box
            }

        adjust_2_candidate_box, key_point = self.stage_three(
            x,
            adjust_candidate_box=adjust_candidate_box,
            cropped_size=stage_three_para.get('cropped_size'),
            batch_size=stage_three_para.get('batch_size'),
            cls_threshold=stage_three_para.get('cls_threshold'),
            nms_threshold=stage_three_para.get('nms_threshold')
        )

        if use_net_type == 'pro':
            return {
                'position': adjust_2_candidate_box,
                'key_point': key_point
            }

    def decode_target(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            'Please do not use decode_target when inference for MTCNNPredictor.'
        )

    def decode_predict(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            'Please do not use decode_predict when inference for MTCNNPredictor.'
        )
