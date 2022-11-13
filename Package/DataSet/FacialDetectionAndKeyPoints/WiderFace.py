from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
from Package.DataSet.ForObjectDetection.VOC_R import XMLTranslate


class WiderFace(Dataset):
    def __init__(
            self
    ):
        pass

    @staticmethod
    def get_all_original_info(
            root: str,
    ) -> dict:
        info_name = os.path.join(
            root,
            'info.json'
        )
        if os.path.exists(info_name):
            with open(info_name, mode='r') as f:
                original_info = json.load(f)

            print('\nOriginal info.json exists(from WiderFace)! So we do not process again!')

        else:
            original_info = {
                'position': {}
            }
            for train_or_val_or_test in ['train', 'val', 'test']:
                now_set_path = os.path.join(
                    root,
                    train_or_val_or_test
                )
                annotation_path = os.path.join(
                    now_set_path,
                    'Annotations'
                )
                images_path = os.path.join(
                    now_set_path,
                    'JPEGImages'
                )

                annotation_name_vec = os.listdir(annotation_path)
                for annotation_name in tqdm(
                        annotation_name_vec,
                        desc='{}_annotation_process'.format(train_or_val_or_test),
                        position=0
                ):
                    now_xml_trans = XMLTranslate(
                        now_set_path,
                        annotation_name
                    )
                    now_image_abs_path = os.path.join(
                        images_path,
                        now_xml_trans.img_file_name
                    )
                    now_position_vec = [[*obj[1:]] for obj in now_xml_trans.objects]
                    original_info['position'][now_image_abs_path] = now_position_vec

            with open(info_name, mode='w') as f:
                json.dump(original_info, f)
            print('\nSave original info.json successfully!')
        return original_info

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
