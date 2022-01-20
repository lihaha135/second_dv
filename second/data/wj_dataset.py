#WjDataset
import numpy as np
import _pickle as pickle
from pathlib import Path
from second.data.dataset import Dataset, register_dataset
from .eval import get_official_eval_result
import pdb


NumPointFeatures = 4
@register_dataset
class WjDataset(Dataset):

    NumPointFeatures = NumPointFeatures
    '''
    0-pedestrian
    1-unkonw
    2-bicycle
    3-tricycle
    4-car
    5-van
    6-truck
    7-tool_cart
    8-bus
    9-semitrailer
    '''
    def __init__(
        self,
            root_path,
            info_path,
            training=None,
            class_names=None,
            prep_func=None,
            num_point_features=None):
        self._root_path = root_path
        print('WjDataset', info_path)
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self.NumPointFeatures = NumPointFeatures
        if training:  # if training
            self.frac = int(len(infos) * 0.25)
            print("train_dataset_num:", len(infos))

            _cls_infos = {name: [] for name in class_names}
            for info in infos:
                for name in set(info["gt_names"]):
                # for name in set(info["annos"]["name"]):
                    if name in class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

            self._nusc_infos = []

            frac = 1.0 / len(class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                # for name in set(info["annos"]["name"]):
                    if name in class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(infos, dict):
                self._nusc_infos = []
                for v in infos.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = infos
        self._wjdata_infos = self._nusc_infos
        #print(len(self._wjdata_infos))

        # self._wjdata_infos = infos

        # self._class_names = class_names
        #self._class_names = ['car', 'bicycle', 'bus', 'tricycle', 'pedestrian', 'van', 'truck', 'tool_cart', 'semitrailer']
        self._class_names = [
            'car',
            'bicycle',
            'bus',
            'tricycle',
            'pedestrian',
            'semitrailer',
            'truck']
        # self._class_names = ['car', 'bicycle', 'bus', 'motorcycle', 'pedestrian', 'truck']
        self._prep_func = prep_func

        self._cls2label = {}
        self._label2cls = {}
        for i in range(len(self._class_names)):
            self._cls2label[self._class_names[i]] = i
            self._label2cls[i] = self._class_names[i]

    def __len__(self):
        return len(self._wjdata_infos)

    @property
    def ground_truth_annotations(self):
        annos = []
        for i in range(len(self._wjdata_infos)):
            info = self._wjdata_infos[i]
            anno = {}
            gt_boxes = info["gt_boxes"]
            box_num = gt_boxes.shape[0]
            anno["bbox"] = np.zeros((box_num, 4))
            anno["alpha"] = np.zeros(box_num, dtype=np.float32)
            #print(gt_boxes,'8888888')
            anno["location"] = gt_boxes[:, :3]
            anno["dimensions"] = gt_boxes[:, 3:6]
            anno["rotation_y"] = gt_boxes[:, -1]
            anno["name"] = info["gt_names"].tolist()
            anno["gt_labels"] = np.array([self._cls2label[cls] for cls in anno["name"]])
            anno["lidar_path"] = info["lidar_path"]
            annos.append(anno)
        return annos

    def get_sensor_data(self, idx):
        info = self._wjdata_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["lidar_path"]
            },
        }

        # print(info)
        lidar_path = Path(info['lidar_path'])
        #print(lidar_path)
        points = np.fromfile(
                str(lidar_path), dtype=np.float32, count=-1).reshape([-1, NumPointFeatures])
        # print('*********************',points.shape)
        res["lidar"]["points"] = points
        if 'gt_boxes' in info:
            gt_boxes = info["gt_boxes"]
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': info["gt_names"],
            }

        return res

    def __getitem__(self, idx):
        input_dict=self.get_sensor_data(idx)
        # points = input_dict[]
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)

        return example

    def convert_detection_to_wjdata_annos(self, detections):
        annos = []
        #pdb.set_trace()
        label_name=['car','bicycle', 'bus', 'tricycle', 'pedestrian','semitrailer','truck']
        for detection in detections:
            anno = {}
            dt_boxes = detection["box3d_lidar"].detach().cpu().numpy()
            box_num = dt_boxes.shape[0]
            labels = detection["label_preds"].detach().cpu().numpy()
            scores = detection["scores"].detach().cpu().numpy()
            anno["score"] = scores
            anno["bbox"] = np.zeros((box_num, 4))
            anno["alpha"] = np.zeros(box_num, dtype=np.float32)
            anno["dimensions"] = dt_boxes[:, 3:6]
            anno["location"] = dt_boxes[:, :3]
            anno["rotation_y"] = dt_boxes[:, -1]
            anno["name"] = [label_name[int(label)] for label in labels]
            anno["lidar_path"] = detection["metadata"]["token"]
            #'metadata': {'token': '/data/second.pytorch/wj_data/testing/bin/20200903104908_000650.bin'}
            # anno["gt_labels"] = np.array([self._cls2label[cls] for cls in anno["name"]])
            annos.append(anno)

        return annos
    #
    # def evaluation(self, detections, output_dir=None):
    #     gt_annos = self.ground_truth_annotations
    #     dt_annos = self.convert_detection_to_wjdata_annos(detections)
    #
    #     print(gt_annos)
    #
    #
    #     # dt_annos = gt_annos.copy()
    #     _class_names = ['pedestrian',
    #                     'bicycle',
    #                     'car',
    #                     'truck',
    #                     'bus',
    #                     ]
    #     result_wj_dict = get_wjdata_eval_result(gt_annos, dt_annos, _class_names)
    #     return result_wj_dict
    def convert_gt_to_de(self, detections):
        annos = []
        for detection in detections:
            anno = {}
            # print(detection["name"])
            anno["score"] = [0.999]*len(detection["name"])
            anno["bbox"] = detection["bbox"]
            anno["alpha"] = detection["alpha"]
            anno["dimensions"] = detection["dimensions"]
            anno["location"] = detection["location"]
            anno["rotation_y"] = detection["rotation_y"]
            anno["name"] = detection["name"]
            # anno["gt_labels"] = np.array([self._cls2label[cls] for cls in anno["name"]])
            annos.append(anno)

        return annos
    def evaluation(self, detections, output_dir):
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        gt_annos = self.ground_truth_annotations
        #print(gt_annos,'___________________---888888----________________')
       
        dt_annos = self.convert_detection_to_wjdata_annos(detections)
        """
        for i in range(len(dt_annos)):
            if dt_annos[i]["lidar_path"] == gt_annos[i]["lidar_path"]:
                print("true")
            else:
                print("===================================== False")"""
        _class_names =  ['car', 'bicycle', 'bus', 'tricycle', 'pedestrian','semitrailer','truck']
        # _class_names = ['car']
        #result_official_dict = get_official_eval_result_v2(
        #    gt_annos,
        #    dt_annos,
        #    _class_names,
        #    z_axis=z_axis,
        #    z_center=z_center)
        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
             _class_names)
        # print(result_official_dict)
        return result_official_dict
