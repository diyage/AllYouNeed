from Package.BaseDev.tools import BaseTool
import torch


class DevTool(BaseTool):
    def __init__(self):
        pass

    @staticmethod
    def make_target(
            batch_size: int,
            is_real_image: bool,
            *args,
            **kwargs
    ) -> torch.Tensor:
        if is_real_image:
            res = torch.ones(size=(batch_size, ), dtype=torch.float32)

        else:
            res = torch.zeros(size=(batch_size, ), dtype=torch.float32)
        return res

    @staticmethod
    def split_target(
            *args,
            **kwargs
    ):
        raise RuntimeError(
            "we do not need this method(split_target)!"
        )

    @staticmethod
    def split_predict(
            *args,
            **kwargs
    ):
        raise RuntimeError(
            "we do not need this method(split_predict)!"
        )
