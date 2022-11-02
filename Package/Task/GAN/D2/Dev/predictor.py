from Package.BaseDev import BasePredictor


class DevPredictor(BasePredictor):
    def __init__(self):
        super().__init__()

    def decode_target(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            "Please do not use decode_target(for xxPredictor)."
        )

    def decode_predict(
            self,
            *args,
            **kwargs
    ):
        raise RuntimeError(
            "Please do not use decode_predict(for xxPredictor)."
        )
