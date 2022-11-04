from ..Dev import DevPredictor
from .Model import MTCNNModel
import typing
import numpy as np


class MTCNNPredictor(DevPredictor):
    def __init__(
            self,
            model: MTCNNModel
    ):
        super().__init__()
        self.model = model

    def stage_one(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        return candidate_box (after nms)
        """
        pass

    def stage_two(
            self,
            x: np.ndarray,
            candidate_box: np.ndarray
    ) -> np.ndarray:
        """
            return adjust_candidate_box (after nms)
        """
        pass

    def stage_three(
            self,
            x: np.ndarray,
            adjust_candidate_box: np.ndarray
    ) -> typing.Union[np.ndarray, np.ndarray]:
        """
            return adjust_2_candidate_box (after nms) and key-point
        """
        pass

    def detect(
            self,
            x: np.ndarray,
            use_net_type: str,
    ) -> typing.Dict:
        self.model.eval()
        assert use_net_type in ['p', 'pr', 'pro']
        """
        If you want to generate cropped images for training R-Net, please
        set use_net_type as 'p', it will return the candidate_box.
        
        If you want to generate cropped images for training O-Net, please
        set use_net_type as 'pr', it will return the adjust_candidate_box.
        
        If you want to predict an image, please set use_net_type as 'pro'.
        """
        assert len(x.shape) == 3  # just one image

        candidate_box = self.stage_one(x)

        if use_net_type == 'p':
            return {
                'position': candidate_box
            }

        adjust_candidate_box: np.ndarray = None
        if use_net_type == 'pr':
            adjust_candidate_box = self.stage_two(x, candidate_box)
            return {
                'position': adjust_candidate_box
            }

        if use_net_type == 'pro':
            adjust_2_candidate_box, key_point = self.stage_three(x, adjust_candidate_box)
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
            'Please not use decode_target when inference for MTCNNPredictor.'
        )

    def decode_predict(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            'Please not use decode_predict when inference for MTCNNPredictor.'
        )
