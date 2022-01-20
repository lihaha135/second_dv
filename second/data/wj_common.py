import shutil

import numpy as np
from pathlib import Path
import os
import pickle
import csv
import math
import open3d as o3d
import time
import second.core.box_np_ops as box_np_ops

PI_rads = math.pi / 180
import random

class_names = [
    'car',
    'bicycle',
    'bus',
    'tricycle',
    'pedestrian',
    'semitrailer',
    'truck'
]

# 第一列为2021标签类别，第二列为2020标签类别
dic = {
    "0": "0",
    "1": "1",
    "2": "2",
    "4": "10",
    "5": "3",
    "6": "4",
    "8": "5",
    "10": "11",
    "11": "11",
    "12": "8",
    "13": "9",
    "22": "7",
    "23": "12"
}


def custom_draw_geometry(pcd, linesets):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for i in linesets:
        vis.add_geometry(i)
    # vis.add_geometry(linesets)
    render_option = vis.get_render_option()
    render_option.point_size = 0.01
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


def create_wjdata_infos(root_path):
    ############初始化open3D#########################3
    label_path = os.path.join(root_path, 'csv_error')
    data_path = os.path.join(root_path, 'bin')
    labelfiles = os.listdir(label_path)

    val_ratio = 0.1


    train_infos = []
    val_infos = []

    num_nomatch = 0
    for i in range(1, 10000):

        labelfile = label_path + "/" + str(i).zfill(5) + ".csv_error"

        infos = {}
        gt_names = []
        gt_boxes = []

        label_fpath = os.path.join(label_path, labelfile)
        infos['lidar_path'] = data_path + "/" + str(i).zfill(5) + ".bin"

        if not os.path.exists(infos['lidar_path']):
            num_nomatch += 1
            continue
        points1 = np.fromfile(infos['lidar_path'], dtype=np.float32, count=-1).reshape((-1, 4))
        with open(labelfile, 'r') as f:
            reader = csv.reader(f)
            num = 0
            print(labelfile)
            for line in reader:


                # gt_boxes.append([float(i[2])/100,float(i[3])/100,float(i[4])/100,float(i[8])/100,float(i[7])/100,float(i[9])/100,(1.57-float(i[6])*PI_rads)])
                gt_boxes.append(
                    [float(line[0]) / 100,
                     float(line[1]) / 100,
                     float(line[2]) / 100,
                     float(line[3]) / 100,
                     float(line[4]) / 100,
                     float(line[5]) / 100,
                     (1.57 - float(line[6])) + math.pi
                     ]
                )

            infos['gt_boxes'] = np.array(gt_boxes)

            vision_mode = 1
            if vision_mode == 1:
                # #####*****  boxes_corners是转换之后的角点信息  *****#####
                boxes_corners = box_np_ops.center_to_corner_box3d(
                    infos['gt_boxes'][:, :3],
                    infos['gt_boxes'][:, 3:6],
                    infos['gt_boxes'][:, 6],
                    origin=[0.5, 0.5, 0.5],
                    axis=2)

                linesets = []
                for j in range(boxes_corners.shape[0]):
                    points_box = boxes_corners[j]

                    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                          [0, 4], [1, 5], [2, 6], [3, 7]])
                    colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
                    colors[0, :] = np.array([1, 0, 0])
                    colors[4, :] = np.array([1, 0, 0])
                    colors[8, :] = np.array([1, 0, 0])
                    colors[9, :] = np.array([1, 0, 0])

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points_box)
                    line_set.lines = o3d.utility.Vector2iVector(lines_box)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    linesets.append(line_set)

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points1[:, :3])
                # linesets.append(point_cloud)
                # o3d.visualization.draw_geometries(linesets)
                custom_draw_geometry(point_cloud, linesets)
    print(f'num_nomatch is {num_nomatch}')


def process():
    files = os.listdir("/home/cxy/Documents/second_kd/double_lidar/test/csv_error/")
    for i in range(len(files)):
        index = int(files[i].split(".")[0])
        # if index > 4000 or index < 720:
        #     continue
        if index < 720:
            f = open("/home/cxy/Documents/second_kd/double_lidar/test/csv_error/" + files[i], "w+")
            for line in open("/home/cxy/Documents/second_kd/double_lidar/test/csv_error/" + files[i]):
                line = line.strip().split(",")
                # line[1] = dic[line[1]]
                line[6] = str((float(line[6]) + 180) % 360)
                f.write(",".join(line) + "\n")
            f.close()


def pcd_to_npy(file_path):
    points = []
    for line in open(file_path):
        line = line.strip().split(" ")
        if len(line) != 4:
            continue
        line = [float(key) for key in line]
        points.append(line)
    points = np.array(points, dtype=np.float32).reshape((-1, 4))
    return points
    # points.tofile("double_lidar_data_with_label/bin2/" + str(i).zfill(6) + ".bin")

def show(root_path):
    for i in range(1, 10001):
        labelfile = root_path + "/csv_error/" + str(i).zfill(5) + ".csv_error"
        pc_path = root_path + "/bin/" + str(i).zfill(5) + ".bin"
        points = np.fromfile(pc_path, dtype=np.float32, count=-1).reshape((-1, 4))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])

        gt_boxes = []
        for line in open(labelfile):
            line = line.strip().split(",")
            gt_boxes.append(
                [float(line[0]),
                 float(line[1]),
                 float(line[2]),
                 float(line[3]),
                 float(line[4]),
                 float(line[5]),
                 1.57 - float(line[6]) + math.pi
                 ]
            )
        gt_boxes = np.array(gt_boxes)
        boxes_corners = box_np_ops.center_to_corner_box3d(gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=[0.5, 0.5, 0.5], axis=2)
        linesets = []
        for j in range(boxes_corners.shape[0]):
            points_box = boxes_corners[j]

            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
            colors[0, :] = np.array([1, 0, 0])
            colors[4, :] = np.array([1, 0, 0])
            colors[8, :] = np.array([1, 0, 0])
            colors[9, :] = np.array([1, 0, 0])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_box)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            linesets.append(line_set)
        custom_draw_geometry(point_cloud, linesets)

if __name__ == "__main__":
    # root_path=r'/data/32_data_label'
    root_path = '/data/Documents/OpenPCDet/data/Deeproute_open_dataset/training'
    # create_wjdata_infos(root_path)
    show(root_path)


