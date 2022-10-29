from Package.Task.FacialRecognition.D2.Dev import DevTool
import torch


class ArcFaceTool(DevTool):
    def __init__(self):
        super().__init__()

    @staticmethod
    def make_target(
            target: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param target is from data loader(torch)...
        classification task, this method need to do nothing.
        """
        return target

    @staticmethod
    def split_predict(
            predict: dict,
    ) -> dict:
        """
        :param predict is output of model(we design)
        classification task, this method need to do nothing.
        :return:
        """
        return predict

    @staticmethod
    def split_target(
            target: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param target is from data loader(torch)...
        classification task, this method need to do nothing.
        """
        return target
