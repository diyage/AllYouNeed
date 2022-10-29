"""
This packet(predictor) is core of this package.
It will be used in inference phase for decoding ground-truth(target)/model-output(predict).

"""
from abc import abstractmethod


class BasePredictor:
    def __init__(
            self,
    ):
        pass

    @abstractmethod
    def decode_target(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def decode_predict(
            self,
            *args,
            **kwargs
    ):
        pass

