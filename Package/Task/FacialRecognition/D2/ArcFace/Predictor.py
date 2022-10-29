from Package.Task.FacialRecognition.D2.Dev import DevPredictor, DevTool
import torch


class ArcFacePredictor(DevPredictor):
    def __init__(
            self,
            use_l2_norm_feature: bool = False,
    ):
        super().__init__()
        self.use_l2_norm_feature = use_l2_norm_feature

    def decode_predict(
            self,
            predict: dict,
            *args,
            **kwargs
    ) -> dict:
        predict['feature'] = self.decode_pre_feature(
            predict['feature']
        )
        return predict

    def decode_pre_feature(
            self,
            pre_feature: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        """

        :param pre_feature has not been l2_norm(we implement it in ArcFaceHead)
        Do we need output feature(that has been l2_norm)?

        """
        if self.use_l2_norm_feature:
            return DevTool.l2_norm(pre_feature, dim=1)
        else:
            return pre_feature
