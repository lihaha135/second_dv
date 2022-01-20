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
o3d.geometry.AxisAlignedBoundingBox
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
ANCHOR_SIZE = {
    "car" : [],
    "bicycle" : [],
    "bus" : [],
    "tricycle" : [],
    "pedestrian" : [],
    "semitrailer" : [],
    "truck" : []
}
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
    label_path = os.path.join(root_path, 'csv')
    data_path = os.path.join(root_path, 'bin')
    labelfiles = os.listdir(label_path)

    val_ratio = 0.1


    train_infos = []
    val_infos = []

    num_nomatch = 0
    for i in range(len(labelfiles)):

        labelfile = labelfiles[i]

        infos = {}
        gt_names = []
        gt_boxes = []

        label_fpath = os.path.join(label_path, labelfile)
        infos['lidar_path'] = data_path + '/' + labelfile.split('.')[0].split('Radar_')[-1] + '.bin'

        if not os.path.exists(infos['lidar_path']):
            num_nomatch += 1
            continue
        points1 = np.fromfile(infos['lidar_path'], dtype=np.float32, count=-1).reshape((-1, 4))
        with open(label_fpath, 'r') as f:
            reader = csv.reader(f)
            num = 0
            # class_names = [
            #     'car',         0      [4,5, 6]
            #     'bicycle',     1      [2, 10]
            #     'bus',         2      [8, 12]
            #     'tricycle',    3      [3]
            #     'pedestrian',  4      [0]
            #     'semitrailer', 5      [9]
            #     'truck']       6      [ 7, 11]

            print(label_fpath)
            for line in reader:
                if int(line[1]) not in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    continue

                bbox = [float(line[2]) / 100,
                        float(line[3]) / 100,
                        float(line[4]) / 100,
                        float(line[8]) / 100,
                        float(line[7]) / 100,
                        float(line[9]) / 100,
                        # float(line[6]) * PI_rads - math.pi / 2
                        (math.pi / 2 - float(line[6]) * PI_rads) + math.pi
                     ]
                gt_boxes.append(bbox)

                # #####*****  gt_names存储的是标注的名字信息  *****#####
                if int(line[1]) in [0]:
                    gt_names.append(class_names[int(4)])    # pedestrian
                    ANCHOR_SIZE["pedestrian"].append(bbox[3:6])
                elif int(line[1]) in [2, 10]:
                    gt_names.append(class_names[int(1)])    # bicycle
                    ANCHOR_SIZE["bicycle"].append(bbox[3:6])
                elif int(line[1]) in [4, 5, 6]:
                    gt_names.append(class_names[int(0)])    # car
                    ANCHOR_SIZE["car"].append(bbox[3:6])
                elif int(line[1]) in [3]:
                    gt_names.append(class_names[int(3)])    # tricycle
                    ANCHOR_SIZE["tricycle"].append(bbox[3:6])
                elif int(line[1]) in [11, 7]:
                    gt_names.append(class_names[int(6)])    # truck
                    ANCHOR_SIZE["truck"].append(bbox[3:6])
                elif int(line[1]) in [8, 12]:
                    gt_names.append(class_names[int(2)])    # bus
                    ANCHOR_SIZE["bus"].append(bbox[3:6])
                elif int(line[1]) in [9]:
                    gt_names.append(class_names[int(5)])    # semitrailer
                    ANCHOR_SIZE["semitrailer"].append(bbox[3:6])
                if len(gt_boxes) != len(gt_names):
                    print("=========================== error")

                num += 1
            # print(num)
            if num == 0:
                print(label_fpath)
            else:
                infos['gt_boxes'] = np.array(gt_boxes)
                infos['gt_names'] = np.array(gt_names)
            # if i % (int(1 / val_ratio)) == 0:
            #     val_infos.append(infos)
            # else:
            train_infos.append(infos)
            ######split train_data and val_data############
            vision_mode = 0
            if vision_mode == 1:
                # #####*****  boxes_corners是转换之后的角点信息  *****#####
                boxes_corners = box_np_ops.center_to_corner_box3d(
                    infos['gt_boxes'][:, :3],
                    infos['gt_boxes'][:, 3:6],
                    infos['gt_boxes'][:, 6],
                    origin=[0.5, 0.5, 0.5],
                    axis=2)

                linesets = []
                for i in range(boxes_corners.shape[0]):
                    points_box = boxes_corners[i]

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
    for key, val in ANCHOR_SIZE.items():
        val = np.array(val).reshape((-1, 3))
        print("anchor " + key + " : ", np.mean(val, 0))


    # with open(root_path + "/wjdata_info_val.pkl", "wb") as f:
    #     pickle.dump(val_infos, f)
    with open(root_path + "/wjdata_info_train.pkl", "wb") as f:
        pickle.dump(train_infos, f)


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


def label_2021_to_2020(path, index=0):
    start_index, end_index = 0, 386
    csv_path = os.path.join(path, "csv_error")
    bin_path = os.path.join(path, "pcd")

    # save_bin = os.path.join(path, "bin2")
    # if(not os.path.exists(save_bin)):
    #     os.mkdir(save_bin)
    save_csv = os.path.join(path, "csv_error")
    if (not os.path.exists(save_csv)):
        os.mkdir(save_csv)

    # pcd -> bin
    for i in range(start_index, end_index + 1):
        pcd_path = os.path.join(bin_path, str(i) + ".pcd")
        bin_save_path = os.path.join(bin_path, str(i + index).zfill(6) + ".bin")
        points = pcd_to_npy(pcd_path)
        print(pcd_path + "   ==>   ", points.shape[0])
        points.tofile(bin_save_path)
        os.remove(pcd_path)

    # 生成新的csv ： 增加第一列，修改类别对应关系，修改角度
    for i in range(start_index, end_index + 1):
        old_csv_path = os.path.join(csv_path, str(i) + ".csv_error")
        new_csv_path = open(os.path.join(save_csv, str(i + index).zfill(6) + ".csv_error"), "w+")
        print(old_csv_path)
        for line in open(old_csv_path):
            line = line.strip().split(",")
            line[1] = dic[line[1]]
            line[6] = str(360.0 - float(line[6]))
            # s = "3180AE4G" + "," + ",".join(line) + "\n"
            s = ",".join(line) + "\n"
            new_csv_path.write(s)
        new_csv_path.close()
        os.remove(old_csv_path)

    os.rmdir(csv_path)
    os.rename(save_csv, csv_path)

    train_index = int(start_index + (end_index - start_index) * (1 / 7)) + index

    train_path = os.path.join(path, "train")
    if (not os.path.exists(train_path)):
        os.mkdir(train_path)
    os.mkdir(os.path.join(train_path, "bin"))
    os.mkdir(os.path.join(train_path, "csv_error"))
    train_bin = os.path.join(train_path, "bin")
    train_csv = os.path.join(train_path, "csv_error")

    test_path = os.path.join(path, "test")
    if (not os.path.exists(test_path)):
        os.mkdir(test_path)
    os.mkdir(os.path.join(test_path, "bin"))
    os.mkdir(os.path.join(test_path, "csv_error"))
    test_bin = os.path.join(test_path, "bin")
    test_csv = os.path.join(test_path, "csv_error")

    for i in range(start_index + index, end_index + 1 + index):
        if i < train_index:
            csv_ = os.path.join(csv_path, str(i).zfill(6) + ".csv_error")
            new_csv_ = os.path.join(train_csv, str(i).zfill(6) + ".csv_error")
            shutil.move(csv_, new_csv_)

            bin_ = os.path.join(bin_path, str(i).zfill(6) + ".bin")
            new_bin_ = os.path.join(train_bin, str(i).zfill(6) + ".bin")
            shutil.move(bin_, new_bin_)
        else:
            csv_ = os.path.join(csv_path, str(i).zfill(6) + ".csv_error")
            new_csv_ = os.path.join(test_csv, str(i).zfill(6) + ".csv_error")
            shutil.move(csv_, new_csv_)

            bin_ = os.path.join(bin_path, str(i).zfill(6) + ".bin")
            new_bin_ = os.path.join(test_bin, str(i).zfill(6) + ".bin")
            shutil.move(bin_, new_bin_)

    os.rmdir(csv_path)
    os.rmdir(bin_path)
    print("done!")


"""
2020年的720帧数据: 0    - 720
20210831102942 : 800  - 1200
20210831104348 : 1200 - 1500
20210831105051 : 1600 - 2000
20210831105753 : 2000 - 2400
20210831110456 : 2400 - 2800
200210831112007 : 2800 - 3200
20210831113412 : 3200 - 3600
20211020120700-1 : 5000 - 5200
"""

if __name__ == "__main__":
    # root_path=r'/data/32_data_label'
    # root_path = '/data/Documents/second_kd/haidian/test/'
    root_path = "/data/second_dv/datasets/gate8_double/train"
    create_wjdata_infos(root_path)
    # label_2021_to_2020('/data/Documents/second_kd/haidian/20211020131400-1-1/', 768)
    # process()


