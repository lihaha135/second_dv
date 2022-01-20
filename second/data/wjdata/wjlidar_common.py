import numpy as np
from pathlib import Path
import os
import pickle
import csv
import math
import open3d as o3d
import time
import second.core.box_np_ops as box_np_ops
PI_rads=math.pi/180

def custom_draw_geometry(pcd,linesets):
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
    # print(root_path)
    class_names = [
    'car',
    'bicycle',
    'bus',
    'tricycle',
    'pedestrian',
    'semitrailer',
    'truck']
    vision_mode = 0
    label_path = os.path.join(root_path, '1_csv/pointcloud')
    data_path = os.path.join(root_path, '1_bin')
    labelfiles=os.listdir(label_path)
    wjdata_infos=[]
    num_nomatch=0
    for labelfile in labelfiles:
        #print(labelfile)
        # #####***** labelfile是读取的文件夹中所有文件的名字  标注文件 *****#####
        infos={}
        gt_names=[]
        gt_boxes=[]
        # #####***** label_fpath应该是标注文件的路径 *****#####
        label_fpath=os.path.join(label_path,labelfile)
        # print(label_fpath)
        # #####***** 根据读取的标注文件的名字找对应的原始点云文件 *****#####
        infos['lidar_path'] = data_path+'/'+labelfile.split('.')[0].split('Radar_')[-1]+'.bin'
        #print(infos)
        # infos['lidar_path'] = data_path + '/' + labelfile.split('.')[0] + '.bin'
        if not os.path.exists(infos['lidar_path']):
            num_nomatch+=1
            #print(label_fpath)
            continue
        # #####*****  points1是读取的原始点云  *****#####
        # np.fromfile(infos['lidar_path'], dtype=np.float32, count=-1)print(np.fromfile(infos['lidar_path'], dtype=np.float32, count=-1))
        points1 = np.fromfile(infos['lidar_path'], dtype=np.float32, count=-1).reshape((-1, 4))
        #print('points',points1.shape)
        with open(label_fpath,'r') as f:
            # #####***** 遍历csv文件的行 *****######
            reader = csv.reader(f)
            num = 0
            for i in reader:
                if int(i[1]) == 1:
                    continue
                if int(i[1]) == 14:
                    continue
                # #####*****  gt_boxes存储的是标注的的信息，包括质心，长宽高和航向角  *****#####
                gt_boxes.append([float(i[2])/100,float(i[3])/100,float(i[4])/100,float(i[8])/100,float(i[7])/100,float(i[9])/100,(1.57-float(i[6])*PI_rads)+math.pi])
                # gt_boxes.append([float(i[2]),float(i[3]),float(i[4]),float(i[8]),float(i[7]),float(i[9]),math.pi-float(i[6])])
                # #####*****  gt_names存储的是标注的名字信息  *****#####
                if int(i[1]) == 0:
                    gt_names.append(class_names[int(4)])
                elif int(i[1]) == 2 or int(i[1])==10:
                    gt_names.append(class_names[int(1)])
                elif int(i[1]) == 4:
                    gt_names.append(class_names[int(0)])
                elif int(i[1]) == 3:
                    gt_names.append(class_names[int(3)])
                elif int(i[1]) == 7 or int(i[1])==11:
                    gt_names.append(class_names[int(6)])
                elif int(i[1]) == 9:
                    gt_names.append(class_names[int(5)])
                elif int(i[1]) == 8 or int(i[1]) == 12:
                    gt_names.append(class_names[int(2)])
                else:
                    continue
                #   gt_names.append(class_names[int(1)])
                num+=1
            # print(num)
            if num == 0:
                print(label_fpath)
        infos['gt_boxes']=np.array(gt_boxes)
        infos['gt_names']=np.array(gt_names)
        wjdata_infos.append(infos)
        # print('----------------')
        if vision_mode==1:
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
            point_cloud.points = o3d.utility.Vector3dVector(points1[:,:3])
            # linesets.append(point_cloud)
            # o3d.visualization.draw_geometries(linesets)
            custom_draw_geometry(point_cloud,linesets)

    ######split train_data and val_data############
    # print(wjdata_infos)
    print(f'num_nomatch is {num_nomatch}')
    # train_infos=wjdata_infos[:-10]
    # val_infos=wjdata_infos[-10:]
    val_infos=wjdata_infos

    # print(f"train sample: {len(train_infos)}")
    # print(f"val sample: {len(val_infos)}")
    # with open(root_path + "/wjdata_info_train.pkl", "wb") as f:
    #     pickle.dump(train_infos, f)
    with open(root_path + "/wjdata_info_train.pkl", "wb") as f:
        pickle.dump(val_infos, f)

# print(__name__)
if __name__=="__main__":
    # root_path=r'/data/32_data_label'
    root_path = '/data/suan_fa/wj_s_data/training/'
    create_wjdata_infos(root_path)
