from Package.Task.FacialRecognition.D2.Dev import DevEvaluator
from .Model import ArcFaceModel
from .Predictor import ArcFacePredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np


class ArcFaceEvaluator(DevEvaluator):
    def __init__(
            self,
            model: ArcFaceModel,
            predictor: ArcFacePredictor,
            compute_threshold_num: int = 1000,
            distance_type: str = 'cosine_similarity'
    ):
        super().__init__(
            model,
            predictor,
            compute_threshold_num,
            distance_type
        )

    def get_images_path_to_feature(
            self,
            data_loader_test: DataLoader,
            desc: str = 'get_images_path_to_feature',
    ):
        with torch.no_grad():
            self.model.eval()
            res = {}
            for _, (images, image_paths) in enumerate(
                    tqdm(
                        data_loader_test,
                        desc=desc,
                        position=0)
            ):
                images: torch.Tensor = images.to(self.device)

                predict: dict = self.model(images)
                predict = self.predictor.decode_predict(predict)

                features: torch.Tensor = predict.get('feature')
                features: np.ndarray = features.cpu().detach().numpy()

                for img_ind in range(images.shape[0]):
                    res[image_paths[img_ind]] = features[img_ind]
            return res
