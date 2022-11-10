from Package.BaseDev.evaluator import BaseEvaluator
from Package.Task.FacialRecognition.D2.Dev.model import DevModel
from Package.Task.FacialRecognition.D2.Dev.predictor import DevPredictor
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from abc import abstractmethod
from typing import List, Tuple
DATA_PAIR_TYPE = List[List]


class DevEvaluator(BaseEvaluator):
    def __init__(
            self,
            model: DevModel,
            predictor: DevPredictor,
            compute_threshold_num: int = 1000,
            distance_type: str = 'cosine_similarity'
    ):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device
        self.predictor = predictor
        self.distance_type = distance_type
        self.compute_threshold_num = compute_threshold_num

    @abstractmethod
    def get_images_path_to_feature(
            self,
            data_loader_test: DataLoader,
            desc: str = 'get_images_path_to_feature',
    ) -> dict:
        pass

    @staticmethod
    def split_ten_fold(
            distance: np.ndarray,
            is_same_id: np.ndarray,
    ) -> Tuple:
        m = distance.shape[0]

        random_index = np.random.choice(list(range(m)), size=(m,), replace=False)
        cut_ind = int(m * 0.9)

        distance_threshold = distance[random_index[:cut_ind]]
        is_same_id_threshold = is_same_id[random_index[:cut_ind]]

        distance_acc = distance[random_index[cut_ind:]]
        is_same_id_acc = is_same_id[random_index[cut_ind:]]

        return distance_threshold, is_same_id_threshold, distance_acc, is_same_id_acc

    def get_best_threshold(
            self,
            distance: np.ndarray,
            is_same_id: np.ndarray,
    ) -> float:
        max_dis = np.max(distance)
        min_dis = np.min(distance)
        threshold_vec: List[float] = np.linspace(min_dis, max_dis, num=self.compute_threshold_num).tolist()
        acc_vec = []
        for threshold in threshold_vec:
            right_num_a = (distance < threshold).astype(np.float32) * (is_same_id == 1.0).astype(np.float32)
            right_num_b = (distance >= threshold).astype(np.float32) * (is_same_id == 0.0).astype(np.float32)
            right_num = right_num_a.sum() + right_num_b.sum()
            acc = 1.0 * right_num / distance.shape[0]
            acc_vec.append(acc)
        return threshold_vec[np.argmax(acc_vec)]

    @staticmethod
    def get_accuracy_use_threshold(
            distance: np.ndarray,
            is_same_id: np.ndarray,
            threshold
    ) -> float:
        right_num_a = (distance < threshold).astype(np.float32) * (is_same_id == 1.0).astype(np.float32)
        right_num_b = (distance >= threshold).astype(np.float32) * (is_same_id == 0.0).astype(np.float32)
        right_num = right_num_a.sum() + right_num_b.sum()
        acc = 1.0 * right_num / distance.shape[0]
        return acc

    def distance_func(
            self,
            feature_a: np.ndarray,
            feature_b: np.ndarray
    ) -> np.ndarray:
        if self.distance_type == 'cosine_similarity':
            a = torch.from_numpy(feature_a)
            b = torch.from_numpy(feature_b)
            c = -1.0 * torch.cosine_similarity(a, b, dim=1)
            """
            be careful here!
            we hope distance smaller( cosine_similarity bigger)
            """
            return c.numpy()
        else:
            print('we have not implement other distance func,'
                  'so we just use l2 distance')
            c = np.sum((feature_a - feature_b) ** 2, axis=1)
            return np.sqrt(c)

    def get_distance(
            self,
            image_path_to_feature: dict,
            data_pair: DATA_PAIR_TYPE,
    ) -> Tuple[np.ndarray, np.ndarray]:

        feature_a_vec: List = []
        feature_b_vec: List = []
        is_same_id_vec: List = []

        for img_path_a, img_path_b, is_same_id in data_pair:
            if image_path_to_feature.get(img_path_a) is not None and image_path_to_feature.get(img_path_b) is not None:
                feature_a_vec.append(image_path_to_feature[img_path_a])
                feature_b_vec.append(image_path_to_feature[img_path_b])
                is_same_id_vec.append(is_same_id)

        distance: np.ndarray = self.distance_func(
            np.array(feature_a_vec, dtype=np.float32),
            np.array(feature_b_vec, dtype=np.float32)
        )
        return distance, np.array(is_same_id_vec, dtype=np.float32)

    def eval_verification_accuracy(
            self,
            data_loader_test: DataLoader,
            data_pair: DATA_PAIR_TYPE,
            desc: str = 'eval accuracy',
    ):
        """
        1) compute feature
        2) compute distance
        3) randomly cut 10 parts(9 parts used for computing best threshold,
            the last one used for computing accuracy).
        4) compute best threshold
        5) compute accuracy

        :param data_loader_test:
                    used for computing feature(s) (and stored in one dict).
        :param data_pair:
                    used for computing distance
                    (path0, path1, is_same_id)
        :param desc:
                    used for display info.
        :return:
        """
        print()
        print('Start {}'.format(desc))
        """
        1) compute feature
        """
        image_path_to_feature = self.get_images_path_to_feature(
            data_loader_test
        )
        """
        2) compute distance
        """
        distance, is_same_id = self.get_distance(image_path_to_feature, data_pair)

        """
        3) randomly cut 10 parts
        """
        distance_threshold, is_same_id_threshold, distance_acc, is_same_id_acc = self.split_ten_fold(
            distance,
            is_same_id
        )

        """
        4) compute best threshold
        """
        best_threshold = self.get_best_threshold(
            distance_threshold,
            is_same_id_threshold
        )
        """
        5) compute accuracy
        """
        acc = self.get_accuracy_use_threshold(
            distance_acc,
            is_same_id_acc,
            best_threshold
        )
        info = "evaluation results: \n" + \
               "    distance_type: {} \n" + \
               "    best threshold: {:.4f} \n" + \
               "    accuracy: {:.3%}"
        print()
        print(info.format(self.distance_type, best_threshold, acc))


if __name__ == "__main__":
    from Package.DataSet.ForFacialRecognition.LFW import get_data_pair, LFWDataSet, DataLoader
    import albumentations as alb

    root = '/home/dell/data/DataSet/LFW122x122/images'
    pair_path = '/home/dell/data/DataSet/LFW122x122/pairs.txt'
    data_pair_vec = get_data_pair(
        pair_path,
        root
    )
    trans = alb.Compose([
        alb.Resize(112, 112),
        alb.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    lfw = LFWDataSet(
        root,
        trans,
        train=False
    )
    loader = DataLoader(lfw, batch_size=8, shuffle=True, num_workers=8)


