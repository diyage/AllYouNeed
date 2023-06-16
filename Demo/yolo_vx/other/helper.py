import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Package.Task.ObjectDetection.D2.YOLO.VX import *
from Package.Optimizer.WarmUp import WarmUpOptimizer
from Demo.yolo_vx.other.config import YOLOVXConfig
from PIL import ImageFile
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from Package.DataSet.ForObjectDetection.COCO import COCODataSet, KINDS_NAME, NAMES_KIND, kinds_name


ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLOVXHelper:
    def __init__(
            self,
            model: YOLOVXModel,
            config: YOLOVXConfig,
            restore_epoch: int = -1
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.config = config

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = YOLOVXTrainer(
            model,
            self.config.lr_mapping,
            self.config.data_config.width_height_center,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            len(self.config.data_config.kinds_name),
            self.config.multi_positives
        )

        self.predictor_for_show = YOLOVXPredictor(
            self.model,
            self.config.show_config.iou_th_for_show,
            self.config.show_config.conf_th_for_show,
            self.config.show_config.score_th_for_show,
            self.config.data_config.image_size
        )

        self.predictor_for_eval = YOLOVXPredictor(
            self.model,
            self.config.eval_config.iou_th_for_eval,
            self.config.eval_config.conf_th_for_eval,
            self.config.eval_config.score_th_for_eval,
            self.config.data_config.image_size
        )

        self.evaluator = YOLOVXEvaluator(
            self.model,
            self.predictor_for_eval,
            len(self.config.data_config.kinds_name)
        )

        self.visualizer = YOLOVXVisualizer(
            self.model,
            self.predictor_for_show,
            self.config.data_config.class_colors,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.multi_positives
        )

    def restore(
            self,
            epoch: int
    ):
        self.restore_epoch = epoch
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
        saved_file_name = '{}/{}.pth'.format(saved_dir, epoch)
        self.model.load_state_dict(
            torch.load(saved_file_name)
        )

    def save(
            self,
            epoch: int
    ):
        # save model
        self.model.eval()
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.model.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            epoch: int
    ):
        with torch.no_grad():
            saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.visualizer.show_detect_results(
                data_loader_test,
                saved_dir,
                desc='[show predict results]'
            )

    def eval_map(
            self,
            data_loader_test: DataLoader,
            iou_th_list: List[float],
    ):
        """

        这个版本完全由我自主撰写的
        使用起来更加灵活
        如果只想 要 AP50 直接设置iou_th_list=[0.5]
        >> 之前版本的eval_map 主要是针对 voc 数据集, 且 AP50的情况

        >> 和 use_coco_tool_eval_map(调用 pycocotool)相比
            这个版本求出的map更低(大概低1%), 可能是因为直接使用积分求面积导致的？
        """
        kind_ind_vec = list(range(len(self.config.data_config.kinds_name)))
        res = self.evaluator.get_all_ap(
            data_loader_test,
            iou_th_list,
            kind_ind_vec
        )

        print("*"*100)
        for iou_th, all_ap in zip(iou_th_list, res):
            now_iou_th_map = sum(all_ap)/len(all_ap)
            print("now_iou_th: {:.3}, map: {:.2%}, each_ap:{}".format(
                iou_th,
                now_iou_th_map,
                all_ap
            ))
        last_ap = np.mean(res)
        print("last map: {:.2%}".format(last_ap))
        print("*" * 100)
        print()

    def use_coco_tool_eval_map(
            self,
            data_loader_test: DataLoader,
            anno_file: str,
            cache_path: str = 'cache/'
    ):
        data_set: COCODataSet = data_loader_test.dataset

        coco_det = []
        self.model.eval()
        device = next(self.model.parameters()).device
        g_ = COCO(annotation_file=anno_file)

        for img_id in tqdm(
                g_.getImgIds(),
                desc='detect ing ...',
                position=0
        ):
            info = data_set.img_id_to_info.get('{}'.format(img_id))
            img_path, bbox_cat_id_es = info

            image = cv2.imread(img_path)
            origin_h, origin_w, _ = image.shape

            res = data_set.transform(image=image, bboxes=[])
            trans_image = res.get('image')
            img_tensor = torch.tensor(trans_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            outputs = self.model(img_tensor.to(device))
            kps_vec = self.predictor_for_eval.process_one_predict(outputs[0])

            # 注意 :
            # 1) category_id 和 torch 预测的ind 是不一样的, 在早期VOC数据集上 我没有考虑到这个问题,
            #    因此代码中有些变量的命名忽略了cat_id 和 kind_ind的区别, 请务必注意

            # 2) 预测的kps_vec 是 scale 到 self.config.data_config.image_size 上的,
            #    如果使用官方的接口, 需要scale到原来的尺寸上

            for kps in kps_vec:
                x1, y1, x2, y2 = kps[1]

                x1 = 1.0 * x1 / self.config.data_config.image_size * origin_w
                y1 = 1.0 * y1 / self.config.data_config.image_size * origin_h
                x2 = 1.0 * x2 / self.config.data_config.image_size * origin_w
                y2 = 1.0 * y2 / self.config.data_config.image_size * origin_h

                kind_name = kinds_name[kps[0]]
                category_id = NAMES_KIND[kind_name]

                # print(kps[1], category_id)

                coco_det.append({
                    'score': kps[2],
                    'category_id': category_id,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'image_id': img_id
                })

            # print('*'*100)

        os.makedirs('{}/'.format(cache_path), exist_ok=True)

        with open('{}/pre.json'.format(cache_path), mode='w') as f:
            json.dump(coco_det, f)

        p_ = g_.loadRes('{}/pre.json'.format(cache_path))

        coco_eval = COCOeval(g_, p_, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOVXLoss(
            self.config.train_config.weight_position,
            self.config.train_config.weight_conf_has_obj,
            self.config.train_config.weight_conf_no_obj,
            self.config.train_config.weight_conf_obj,
            self.config.train_config.weight_cls
        )

        sgd_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.train_config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        warm_optimizer = WarmUpOptimizer(
            sgd_optimizer,
            base_lr=self.config.train_config.lr,
            warm_up_epoch=self.config.train_config.reach_base_lr_cost_epoch
        )

        for epoch in tqdm(range(self.restore_epoch+1,
                                self.config.train_config.max_epoch_on_detector),
                          desc='training object detector',
                          position=0):
            torch.cuda.empty_cache()
            loss_dict = self.trainer.train_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                desc='train for detector epoch --> {}'.format(epoch),
                now_epoch=epoch,

            )

            print_info = '\n\nepoch: {} [ now lr:{:.6f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            self.save(epoch)
            self.show_detect_results(data_loader_test, epoch)

            if epoch % self.config.eval_config.eval_frequency == 0:
                self.eval_map(
                    data_loader_test,
                    # iou_th_list=np.arange(0.5, 1.0, step=0.05).tolist()
                    iou_th_list=[0.5]
                )
