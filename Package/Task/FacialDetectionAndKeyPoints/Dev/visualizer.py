from Package.BaseDev import BaseVisualizer
import abc


class DevVisualizer(BaseVisualizer):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def show_result(
            self,
            *args,
            **kwargs
    ):
        raise NotImplementedError(
            "Please implement this method show_result for xxVisualizer."
        )
