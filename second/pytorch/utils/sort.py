# #####*****  策略1:加入了框内框外点的判断，以及新的高速低速判断，而不是采用动静的绝对判断，
# #####*****  策略2:对机动车采用加速模型，对于非机动车采用匀速模型，
# #####*****  策略3:将两个跟踪列表进行合并,统一采用IOU进行数据关联,但是在级联阶段可选IOU和距离两种规则，需要修改传入的参数
# #####*****  策略4:关于历史状态，保证列表里存储的首尾两个状态量的距离在3m左右
# #####*****  策略5:静止目标框容易出现飘框，考虑到低速目标其运动状态基本不变，尝试其不进行卡尔曼的预测步骤，当有更新的时候执行更新
# #####*****  策略6:对于静态目标，通过增大其卡尔曼中的Q矩阵，来使其状态平滑，减少飘框等现象，功能待测试  当前更新了两个参数self.min_q, self.max_q，
# #####*****  保证参数缩小和增大都只有一次，不会累乘
# #####*****  改进1:将车道信息做成可选择的  加载车道信息文件，读取配置,车道信息的规定是从雷达y轴正轴方向开始，顺时针开始统计
# #####*****  两种计算框内框外的方式修改
# #####*****  尝试不将列表的长度固定了，而是将目标分为高速低速存储不同长度的列表，低速长，高速短,某种意义上来说，其实目标完整的进入场景中，都会保存一个参考的航向角的

# #####*****  尝试1:尝试除去高速和低速目标，再加上动静目标属性,依靠旧的速度和位移共同判断动静的方法

# #####*****  之前写的在数据关联函数中既做IOU匹配又做距离匹配，存在问题,需要按照之前两个跟踪列表的方式才可以
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from numba import jit
import numpy as np
import time
import math
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import  linear_sum_assignment as linear_assignment
from filterpy.kalman import KalmanFilter
import second.core.box_np_ops as box_np_ops
import os
# from CommonDefine import *

Rads_cov = 180 / math.pi


#############对lidar计算iou值##############
def rotate_nms_cc(dets, trackers):
    trackers_corners = box_np_ops.center_to_corner_box2d(trackers[:, :2], trackers[:, 2:4], trackers[:, 4])
    trackers_standup = box_np_ops.corner_to_standup_nd(trackers_corners)
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2:4], dets[:, 4])
    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)
    # standup_iou = box_np_ops.iou_jit(dets_standup, trackers_standup, eps=0.0)
    standup_iou, standup_iou_new = iou_jit_new(dets_standup, trackers_standup, eps=0.0)
    return standup_iou, standup_iou_new


@jit(nopython=True)
def iou_jit_new(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    overlaps_new = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                            (boxes[n, 2] - boxes[n, 0] + eps) *
                            (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
                    overlaps_new[n, k] = iw * ih / box_area
    return overlaps, overlaps_new


def cal_angle(state_list, thresh):
    dis_x = state_list[-1][0][0] - state_list[0][0][0]
    dis_y = state_list[-1][0][1] - state_list[0][0][1]
    dis_len = (dis_x * dis_x + dis_y * dis_y) ** 0.5
    if dis_len > thresh:
        dis_angle = math.acos(dis_x / dis_len) * Rads_cov - 180
        if dis_y > 0:
            dis_angle = 90 - dis_angle
        else:
            dis_angle = 90 - (360 - dis_angle)
        dis_angle = (dis_angle % 360)
    else:
        dis_angle = None
    return dis_angle


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        self.use_acc_model = 0
        if bbox[7] in list([0, 2, 6]):
            # if bbox[7] in list([0, 1, 2, 3, 4, 5, 6]):
            # if bbox[7] in list([10]):
            self.use_acc_model = 1
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.F = np.array([[1, 0, 0, 0, 0.1, 0, 0.005, 0],
                                  [0, 1, 0, 0, 0, 0.1, 0, 0.005],
                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0.1, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0.1],
                                  [0, 0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 1]])
            self.kf.H = np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0]
                 ])
            self.kf.Q *= 0.1
            self.kf.R *= 1
            self.kf.P *= 10
            self.kf.P[4:6, 4:6] *= 1000
        else:
            self.use_acc_model = 0
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 1]])
            self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0],
                                  ])
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.kf.R *= 10
            self.kf.P *= 10
            self.kf.P[4:6, 4:6] *= 1000

        ####使用box_lidar,里面包含中心点，面积和航向角度
        self.kf.x[:4] = bbox[:4].reshape((-1, 1))
        self.bbox = bbox  ###对应存储的状态值
        self.time_since_update = 0

        # #####*****  减小和增大卡尔曼Q矩阵的标志位  *****#####
        self.min_q = True
        self.max_q = False

        # #####*****  存储检测目标的稳定航向角，以车道角和轨迹角为标准
        self.final_angle = None

        # #####*****  self.id 表示检测目标的ID  *****#####
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        # #####*****  存储目标的状态，在预测函数中会输出  *****#####
        self.history = []
        self.head_angle = []
        # #####*****  self.state存储卡尔曼滤波的状态量，目前来看存储的是预测的状态量，有更新时存储的是更新后的状态量(状态添加的位置可变，视情况而定)  *****#####
        self.state = []
        # #####*****  self.label_dict存储的对应的目标的anchor信息  *****#####
        self.label_dict = {0: [1.95017717, 4.60718145, 1.72270761], 1: [0.60058911, 1.68452161, 1.27192197],
                           2: [2.94046906, 11.1885991, 3.47030982],
                           3: [0.76279481, 2.09973778, 1.44403034], 4: [0.66344886, 0.7256437, 1.75748069],
                           5: [0.39694519, 0.40359262, 1.06232151],
                           6: [2.4560939, 6.73778078, 2.73004906]}

        # #####*****  存储检测出来的航向角，保证检测角不跳变  *****#####
        self.angle_list = []

        # #####*****  高速目标  低速目标  *****#####
        self.high_speed = False
        self.low_speed = False

        # #####*****  动态目标  静态目标  *****#####
        self.dynamic = False
        self.static = False

        # #####*****  存储状态量判断目标的动静属性,不使用动静属性的时候可以不用  *****#####
        self.state_judge = []

        # #####*****  车道航向角  轨迹航向角  检测航向角  *****#####
        self.lane_angle = None
        self.track_angle = None
        self.detec_angle = None

        # #####*****  存储的也是状态量，是一个列表，长度在10，在get_state()中会操作输出  *****#####
        self.angle_box = []
        # #####*****  存储历史10帧的类别消息，用途是，寻找每个目标10帧内出现次数最多的标签作为新一帧目标  *****#####
        self.label_box = []
        # #####*****  击中次数 也就是更新的次数  *****#####
        self.hits = 0

        ################追踪目标的速度################
        self.speed = 0
        #####*****存储更改的label信息*****#####
        self.label = 0
        #####*****高速和低速的阈值*****#####
        self.speed_thresh = 3

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        ####表示距离上一次更新后没有再匹配更新的次数
        self.time_since_update = 0
        self.history = []
        self.hits += 1  ###表示在age时间内其能够匹配上的次数
        ####使用匹配到的box信息对当前的卡尔曼状态量进行更新
        self.kf.update(bbox[:4].reshape((-1, 1)))
        # #####*****  更新存储的目标信息  *****#####
        self.bbox = bbox
        # #####*****  该部分只是为了解决新的模型类别个数与旧模型不匹配，确定新模型的时候，可以修改self.label_dict，并去掉这部分
        if self.bbox[7] == 5:
            self.bbox[7] = 4
        self.label_box.append(self.bbox[7])
        ######保证列表的长度，防止长度过长
        if len(self.label_box) > 50:
            self.label_box = self.label_box[-50:]
        ######分析历史帧的标签，取次数最多的label，并赋对应的box尺寸
        if len(self.label_box) > 0:
            more_label = max(self.label_box, key=self.label_box.count)
            self.label = more_label
            # # #####*****  这部分是修改目标的尺寸  *****#####
            # self.bbox[2] = self.label_dict[self.label][0]
            # self.bbox[3] = self.label_dict[self.label][1]
            # self.bbox[6] = self.label_dict[self.label][2]
            # #####*****  满足条件时修改目标状态的标签信息，或者同时修改存储标签信息的列表，保证目标类别不跳变
            if self.label_box.count(self.label) / len(self.label_box) > 0.7:
                self.bbox[7] = self.label
                # if self.label_box.count(self.label) > 35:
                self.label_box[-1] = more_label
        else:
            self.label = self.bbox[7]

        # #####*****  因为加速和匀速模型对于F矩阵的定义不一样，造成速度计算方式不一样，后续可以直接修改F矩阵即可，就不用这部分的判断条件了  *****#####
        if self.use_acc_model:
            self.speed = math.sqrt(self.kf.x[4] ** 2 + self.kf.x[5] ** 2)
        else:
            self.speed = 10 * math.sqrt(self.kf.x[4] ** 2 + self.kf.x[5] ** 2)
        # #####*****  高速低速目标的判别  *****#####
        if self.speed > self.speed_thresh:
            self.high_speed = True
            self.low_speed = False
        else:
            self.high_speed = False
            self.low_speed = True
        # #####*****  判断动静目标的条件，不需要的话直接注释掉  *****#####
        if len(self.state_judge) > 9:
            diff_x = self.state_judge[-1][0] - self.state_judge[0][0]
            diff_y = self.state_judge[-1][1] - self.state_judge[0][1]
            diif_dis = (diff_x ** 2 + diff_y ** 2) ** 0.5
            if self.speed < 3 and diif_dis < 1.5:
                self.static = True
                self.dynamic = False
            else:
                self.static = False
                self.dynamic = True

        # # #####*****  低速的时候，将卡尔曼的Q矩阵变小，让轨迹更平滑，即受离谱检测目标的影响会小些，待测试的功能  *****#####
        # # #####*****  考虑对机动车采用Q矩阵策略 (可以按照动静和高速低速来进行判断的方式) *****#####
        if self.label in [0, 2, 6]:
            # if (not self.high_speed) and self.min_q and self.static:
            if (not self.high_speed) and self.min_q:
                # if self.static and self.min_q:
                self.kf.Q *= 0.1
                self.min_q = False
                self.max_q = True
            if self.high_speed and self.max_q:
                # if self.dynamic and self.max_q:
                self.kf.Q *= 10
                self.min_q = True
                self.max_q = False

        # # #####*****  原始的保证状态列表的长度，根据列表头尾的距离计算轨迹航向角  *****#####
        # self.state.append(self.kf.x[:2, :].reshape(1, -1))
        # if len(self.state) > 10:
        #     self.state = self.state[-10:]

        # #####*****  根据状态列表计算距离，找到距离最接近3m的索引，然后保留该索引之后的所有状态量   *****#####
        # #####*****  在更新函数中添加，存在部分目标间隔好多帧才更新，这就造成两个状态之间的距离较远，超过3m  *****#####
        # #####*****  在预测函数中添加，依然会存在部分这样的问题，只是出现的次数会减少  *****#####
        # #####*****  此处计算距离的方式，占整个更新运行时间的一半  *****#####
        # #####*****  采用原始的保证列表长度的方式，不如保证距离可靠，但是时间会少，此处的改进在于对高低速目标采用不同长度的列表  *****#####
        if self.speed > 0.8:
            self.state.append(self.kf.x[:2, :].reshape(1, -1))
        if self.high_speed:
            if len(self.state) > 10:
                self.state = self.state[-10:]
        else:
            if len(self.state) > 30:
                self.state = self.state[-30:]
        # if len(self.state) > 1:
        #     a = np.asarray(self.state).reshape(-1, 2)
        #     diff = a - self.state[-1].reshape(-1, 2)
        #     eve_dis = abs((diff[:, 0] ** 2 + diff[:, 1] ** 2) ** 0.5).reshape(-1, 1)
        #     eve_dis -= 3
        #     po_indice = np.where(eve_dis > 0)
        #     if po_indice[0].shape[0] > 0:
        #         new_eve_dis = eve_dis[po_indice[0], 0]
        #         min_dis_indice = np.argmin(new_eve_dis)
        #         if eve_dis[po_indice[0][min_dis_indice]] > 0:
        #             self.state = self.state[po_indice[0][min_dis_indice]:]
        # print('lalalala')
        # print(len(self.state))
        # if len(self.state) > 2:
        #     a = np.asarray(self.state).reshape(-1, 2)
        #     diff = a[0] - self.state[-1].reshape(-1, 2)
        #     diff_bak = a[1] - self.state[-1].reshape(-1, 2)
        #     eve_dis = abs((diff[:, 0] ** 2 + diff[:, 1] ** 2) ** 0.5).reshape(-1, 1)
        #     eve_dis_bak = abs((diff_bak[:, 0] ** 2 + diff_bak[:, 1] ** 2) ** 0.5).reshape(-1, 1)
        #     if eve_dis > 3.5 and eve_dis_bak > 3:
        #         self.state = self.state[1:]

        ######self.angle_box存储的是检测状态量
        self.angle_box.append(self.bbox)
        ######保证列表的长度，防止长度过长
        if len(self.angle_box) > 10:
            self.angle_box = self.angle_box[-10:]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # # #####*****  对于速度较小，且连续>=3帧没有更新的轨迹不进行预测  *****#####
        if self.speed > 2 or self.time_since_update < 3:
            self.kf.predict()
        # self.kf.predict()
        self.time_since_update += 1
        ########30##直接使用box_lidar#####不需要进行转换##########
        output_history = self.kf.x[:4].reshape((1, 4))
        self.history.append(output_history)

        # #####*****  这里存储的是判断动静的状态量，在预测中给出，保证状态的连续性  *****#####
        self.state_judge.append(self.kf.x)
        if len(self.state_judge) > 10:
            self.state_judge = self.state_judge[-10:]

        return self.history[-1], self.bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        ##########直接使用box_lidar#####不需要进行转换##########
        output_x = self.kf.x[:4].reshape((1, 4))

        # #####*****  对于速度低的目标做一个平滑  *****#####
        if self.speed < 0.5 and len(self.angle_box) > 1:
            x_mean = np.asarray(self.angle_box)
            output_x = np.mean(x_mean[:, :4], axis=0).reshape((1, 4))
        return output_x, self.bbox, self.speed, self.angle_box


def associate_detections_to_trackers(detections, trackers):
    """
    数据关联的函数，将跟踪轨迹和检测目标关联起来
    Assigns detections to tracked o192.168.3.181bject (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int), 0

    #####直接使用lidar部分的iou，这里直接使用矩阵就行#####
    detections[:, 4] *= (np.pi / 180)
    trackers[:, 4] *= (np.pi / 180)
    # #####*****  rotate_nms_cc()函数在本文件中，进行了部分修改，主要是计算IOU的方式，返回了两个IOU矩阵  *****#####
    # #####*****  重合面积: area  检测框A的面积: area_A  轨迹框B的面积:area_B  *****#####
    # #####*****  iou_matrix = area / (area_A + area_B - area)     iou_matrix_new = area / area_B
    iou_matrix, iou_matrix_new = rotate_nms_cc(detections, trackers)

    # # #####*****  另外一种判断关联目标质心是否在轨迹框内的方法  *****#####
    # big_trackers = trackers
    # # #####*****  1.5是将目标框扩大，考虑稀疏点云检测目标的不稳定性，扩大框的尺寸，增加匹配上的概率，当然，你也可以选择不扩大  *****#####
    # big_thresh = 1.3
    # big_trackers[:, 2] *= big_thresh
    # big_trackers[:, 3] *= big_thresh
    #
    # # #####*****  给出跟踪轨迹的目标的四个角点的信息  *****#####
    # tra_corners = box_np_ops.center_to_corner_box2d(big_trackers[:, :2], big_trackers[:, 2:4], big_trackers[:, 4])
    # detections[:, 4] *= (180 / np.pi)
    # trackers[:, 4] *= (180 / np.pi)

    cost_matrix = iou_matrix
    iou_threshold = 0.000001
    # #####****  匈牙利算法获取匹配的检测和跟踪轨迹  *****#####
    matched_indices = linear_assignment(-cost_matrix)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        # #####*****  以下一部分是为了计算关联点是在框内还是框外,根据向量叉乘计算面积，对比面积即可,类似多边形计算面积  *****#####
        # #####*****  扩大框尺寸之后的长和宽，计算面积  ******#####
        # l = trackers[m[1], 2]*big_thresh
        # w = trackers[m[1], 3]*big_thresh
        # #####*****  tra_corners是计算出来的扩大框之后的角点信息  *****#####
        # asso_tra_corners = tra_corners[m[1]]
        # stand_point = np.array([[detections[m[0]][0], detections[m[0]][1]]])
        # new_diff = asso_tra_corners - stand_point
        # diff_1 = new_diff[0]
        # diff_2 = new_diff[1]
        # diff_3 = new_diff[2]
        # diff_4 = new_diff[3]
        # area_1 = 0.5 * abs(diff_1[0] * diff_2[1] - diff_1[1] * diff_2[0])
        # area_2 = 0.5 * abs(diff_2[0] * diff_3[1] - diff_2[1] * diff_3[0])
        # area_3 = 0.5 * abs(diff_3[0] * diff_4[1] - diff_3[1] * diff_4[0])
        # area_4 = 0.5 * abs(diff_4[0] * diff_1[1] - diff_4[1] * diff_1[0])
        # total_area = area_1 + area_2 + area_3 + area_4
        # area_tra = l * w
        # #####*****  trackers的最后一位存储的是轨迹的time_since_update  *****#####
        # if cost_matrix[m[0], m[1]] < iou_threshold or (total_area > (area_tra + 0.01) and trackers[m[1]][-1] < 3):
        if cost_matrix[m[0], m[1]] < iou_threshold or (iou_matrix_new[m[0], m[1]] < 0.148 and trackers[m[1]][-1] < 3):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), cost_matrix


class Sort(object):
    def __init__(self, max_age=4, min_hits=2):
        """
        跟踪器的初始化
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.max_age_new = 30
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        #####*****判断是否利用位移计算航向角的阈值(暂时觉得速度作为判断条件不太靠谱，采用位移作为判断条件)自己看情况修改吧，可以适当大*****#####
        self.dis_thresh = 3
        # #####*****  角度差，将相邻帧的航向角的变化限制在正负self.angle_judge内  *****#####
        self.angle_judge = 10

    def update(self, dets, line_k, line_b, line_limit, line_angle):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.参数输入为
        x,y,l,角度，z坐标，高度
        line_k: 车道的斜率
        line_b: 车道的截距
        line_limit: 车道限制的截止坐标值
        line_angle: 车道对应的航向角
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may diffkf_speed_xer from the number of detections provided.
        """
        start_time = time.time()
        self.frame_count += 1
        # get predicted locations from existing trackers.
        # trks = np.zeros((len(self.trackers),5))
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos, bbox = self.trackers[t].predict()
            pos = pos[0]
            trk[:] = [pos[0], pos[1], bbox[2], bbox[3], bbox[4], self.trackers[t].time_since_update,
                      self.trackers[t].speed]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        # print('the data prepare time is : ', (time.time() - start_time) * 1000)
        start_time = time.time()
        # #####*****  执行数据关联函数  *****#####
        matched, unmatched_dets, unmatched_trks, cost_matrix = associate_detections_to_trackers(dets, trks)
        # print('the associate time is : ', (time.time() - start_time) * 1000)

        start_time = time.time()
        for t, trk in enumerate(
                self.trackers):  # if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
        # print('the update time is : ', (time.time() - start_time) * 1000)
        start_time = time.time()
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        # print('the new tracker init time is : ', (time.time() - start_time) * 1000)
        start_time = time.time()
        num_tra = len(self.trackers)
        for trk in reversed(self.trackers):
            d, x_temp, trk_speed, trk_angle_box = trk.get_state()
            d = d[0]
            # #####*****  存储检测航向角,并保证列表的长度  *****#####
            trk.angle_list.append((x_temp[4]) % 360)
            if len(trk.angle_list) > 10:
                trk.angle_list = trk.angle_list[-10:]
            # #####*****  保证检测的航向角不跳变  *****#####
            if len(trk.angle_list) > 1:
                angle_diff = abs(trk.angle_list[-1] - trk.angle_list[-2])
                if angle_diff < self.angle_judge or angle_diff > (360 - self.angle_judge):
                    pass
                elif angle_diff > (180 - self.angle_judge) and angle_diff < (180 + self.angle_judge):
                    trk.angle_list[-1] = (trk.angle_list[-1] + 180) % 360
                else:
                    trk.angle_list[-1] = trk.angle_list[-2]
            trk.detec_angle = trk.angle_list[-1]
            # #####*****  根据检测目标的位置，确定车道航向角  *****#####
            # diff_l = d[1] - line_k * d[0] - line_b
            # diff_l1 = diff_l[0][0]
            # diff_l2 = diff_l[0][1]
            # diff_l3 = diff_l[0][2]
            # diff_l4 = diff_l[0][3]
            # diff_l5 = diff_l[0][4]
            # diff_l6 = diff_l[0][5]
            # diff_l7 = diff_l[0][6]
            # diff_l8 = diff_l[0][7]
            # diff_l9 = diff_l[0][8]
            # diff_l10 = diff_l[0][9]
            # diff_l11 = diff_l[0][10]
            # diff_l12 = diff_l[0][11]
            #
            # # #####*****region1_thresh是为了限定路口之外的行车区域，我们在路口中有自定义的航向角策略*****#####
            # if (diff_l1 * diff_l2) < 0 and d[1] > line_limit[0][0]:
            #     trk.lane_angle = line_angle[0][0]
            # elif (diff_l2 * diff_l3) < 0 and d[1] > line_limit[0][0]:
            #     trk.lane_angle = line_angle[0][1]
            # elif (diff_l4 * diff_l5) < 0 and d[0] > line_limit[0][1]:
            #     trk.lane_angle = line_angle[0][2]
            # elif (diff_l5 * diff_l6) < 0 and d[0] > line_limit[0][1]:
            #     trk.lane_angle = line_angle[0][3]
            # elif (diff_l7 * diff_l8) < 0 and d[1] < line_limit[0][2]:
            #     trk.lane_angle = line_angle[0][4]
            # elif (diff_l8 * diff_l9) < 0 and d[1] < line_limit[0][3]:
            #     trk.lane_angle = line_angle[0][5]
            # elif (diff_l10 * diff_l11) < 0 and d[0] < line_limit[0][3]:
            #     trk.lane_angle = line_angle[0][6]
            # elif (diff_l11 * diff_l12) < 0 and d[0] < line_limit[0][3]:
            #     trk.lane_angle = line_angle[0][7]
            # else:
            #     trk.lane_angle = None
            # #####*****  计算目标的轨迹航向角  *****#####
            if len(trk.state) > 1:
                thresh = self.dis_thresh
                trk.track_angle = cal_angle(trk.state, thresh)
            # #####*****  高速目标的航向角优先顺序是轨迹航向角，车道航向角，检测航向角  *****#####
            # #####*****  低速目标的航向角优先顺序是车道航向角，轨迹航向角，检测航向角  *****#####
            if trk.high_speed:
                if trk.track_angle is not None:
                    head_angle = trk.track_angle
                    trk.final_angle = head_angle
                elif trk.lane_angle is not None:
                    head_angle = trk.lane_angle
                    trk.final_angle = head_angle
                else:
                    head_angle = trk.detec_angle
            else:
                if trk.lane_angle is not None:
                    head_angle = trk.lane_angle
                    trk.final_angle = head_angle
                elif trk.track_angle is not None:
                    head_angle = trk.track_angle
                    trk.final_angle = head_angle
                else:
                    head_angle = trk.detec_angle
            # #####*****  保证航向角不发生较大的突变  *****#####
            if trk.final_angle is not None:
                angle_diff = abs(trk.final_angle - head_angle) % 360
                if angle_diff < self.angle_judge or angle_diff > (360 - self.angle_judge):
                    pass
                elif angle_diff > (180 - self.angle_judge) and angle_diff < (180 + self.angle_judge):
                    head_angle = (head_angle + 180) % 360
                else:
                    head_angle = trk.final_angle
            trk.head_angle.append(head_angle)

            if ((trk.time_since_update < self.max_age) and (
                    trk.hits >= self.min_hits or self.frame_count <= self.min_hits)) or (
                    (trk.time_since_update < self.max_age_new) and (trk.hits >= 10)):
                head_final = trk.head_angle[-1]
                # #####*****  丢失次数小于6时候向外输出检测结果  *****#####
                if trk.time_since_update < 6:
                    d_conv = [d[0], d[1], x_temp[2], x_temp[3], trk.bbox[4], x_temp[5], x_temp[6], trk.label]
                    # d_conv = [x_temp[0], x_temp[1], x_temp[2], x_temp[3], x_temp[4]*180/np.pi, x_temp[5], x_temp[6], trk.label]
                    ret.append(np.concatenate(
                        (d_conv, [trk.speed], [trk.id + 1], [x_temp[8]], [trk.hits], [0], [0], [0], [0])).reshape(1,
                                                                                                                  -1))
                    """
                    x, y, w, l, yaw, z, h, cls, speed, id, conf, hit, 0, 0, 0, 0
                    """
            num_tra -= 1

            ##将预测大于长寿命最大值，且命中次数不满足长寿名阈值的删除掉
            # #####*****  丢失次数大于self.max_age，且更新次数小于10,认为目标状态不稳定，不进行长时间跟踪  *****#####
            # #####*****  丢失次数大于self.max_age_new，删除  *****#####

            if (trk.time_since_update > self.max_age) and (trk.hits < 10):
                self.trackers.pop(num_tra)
            elif trk.time_since_update > self.max_age_new:
                self.trackers.pop(num_tra)
        # print('the strategy time is : ', (time.time() - start_time) * 1000)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 16))

if __name__ == "__main__":
    tracker = Sort()
    predict_path = "/media/cxy/测试组-7号盘/20211020000000/res/"
    save_path = "/media/cxy/测试组-7号盘/20211020000000/res_sort/"
    line_k = np.load("/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_k.npy")
    line_b = np.load("/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_b.npy")
    line_limit = np.load("/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_limit.npy")
    line_angle = np.load("/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_angle.npy")

    for i in range(2647):
        with open(os.path.join(predict_path, str(i) + ".csv_error"), encoding='utf-8') as f:
            pre_bbox = np.loadtxt(f, delimiter=",")

        # pre_bbox = np.load(os.path.join(predict_path, str(i) + ".npy"))
        pre_bbox = pre_bbox[:, [0, 1, 3, 4, 6, 2, 5, 7, 8]]
        res = tracker.update(pre_bbox, line_k, line_b, line_limit, line_angle)
        res = res.astype(np.float32)
        np.savetxt(save_path + str(i) + ".csv_error", res, delimiter=",")
        print("The {} frame".format(i))
