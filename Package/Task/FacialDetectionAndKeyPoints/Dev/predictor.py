from Package.BaseDev import BasePredictor
import abc


class DevPredictor(BasePredictor):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def decode_predict(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement this method decode_predict for xxPredictor."
        )

    @abc.abstractmethod
    def decode_target(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement this method decode_target for xxPredictor."
        )
