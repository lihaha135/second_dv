from numba import jit
import os
from PIL import Image
import numpy as np
import math
import time
import dpkt
import cv2
import open3d as o3d

PI_rad = math.pi / 180


def custom_draw_geometry(points):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(point_cloud)
    render_option = vis.get_render_option()
    render_option.point_size = 1
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


# im_size : 2000
def save_bmp(pcData, im_size, bmp_path):
    img_corner = im_size // 2
    dist2pixel = im_size // 200
    # framdata = np.fromfile(filename, dtype=np.float32, count=-1).reshape((-1, 4))[:, :3]

    img = Image.new('RGB', (im_size, im_size), 'black')
    point_num = pcData.shape[0]
    if point_num == 0:
        print('there is no data')
    for i in range(point_num):
        tempx = round(pcData[i, 0] * dist2pixel)
        tempy = round(pcData[i, 1] * dist2pixel)

        if (tempx >= img_corner) or (tempy >= img_corner) or (tempx <= -img_corner) or (tempy <= -img_corner):
            continue
        tempx = int(img_corner - 1 + tempx)
        tempy = int(img_corner - 1 - tempy)

        img.putpixel((tempx, tempy), (255, 255, 0))
    print(img.getpixel((0, 1)))

    img.save(bmp_path)


#
def cache_bmp(img):
    img_convert = np.ones_like(img)

    (height, width, color) = img.shape
    for y in range(height):
        conver_y = height - y - 1
        img_convert[y, :, :] = img[conver_y, :, :]
    return img_convert


@jit(nopython=True)
def crop_pcdata(framdata, img):
    im_size = img.shape[0]
    img_corner = im_size // 2
    dist2pixel = im_size // 200
    points_num = framdata.shape[0]
    in_points = np.zeros((points_num, 4))
    # out_points = np.zeros((int(points_num), 4))
    count_in = 0
    for i in range(points_num):
        tempoint = framdata[i]

        xindex = math.ceil(tempoint[0] * dist2pixel) + img_corner - 1
        lineindex = math.ceil(tempoint[1] * dist2pixel) + img_corner - 1

        if (xindex >= im_size) or (lineindex >= im_size) or (xindex <= 0) or (lineindex <= 0):
            pass
        # elif (img[lineindex, xindex, 0], img[lineindex, xindex, 1], img[lineindex, xindex, 2]) == (255, 255, 255):
        elif img[lineindex, xindex, 2] == 255:
            ###这里因为底色是黑色（0,0,0）,而点云颜色是（255,0,255）绿色为（255,255,0）,擦出后的是白色（255,255,255）
            pass
        else:
            in_points[count_in, :] = tempoint
            count_in += 1
    in_points = in_points[:count_in, :]
    # print('in_points shape',in_points.shape)

    return in_points


# 不用了
# @jit(nopython=True)
def back_image(backdata, back_img, grid_size):
    grid_size_inv = 1 / grid_size
    im_size = round(200 * grid_size_inv)
    min_x, min_y = -100, -100
    img = np.zeros((im_size, im_size, 4))
    backdata = backdata[backdata[:, 2] > -5.5]
    point_num = backdata.shape[0]
    if point_num == 0:
        print('there is no data')
    for i in range(point_num):
        tempx = math.floor((backdata[i, 0] - min_x) * grid_size_inv)
        tempy = math.floor((backdata[i, 1] - min_y) * grid_size_inv)
        tempz = backdata[i, 2]
        if (tempx > im_size - 1) or (tempx < 0) or (tempy > im_size - 1) or (tempy < 0):
            continue
        elif (tempz < img[tempx, tempy, 3]):
            img[tempx, tempy, 3] = tempz

        # elif (back_img[tempx, tempy, 3] == 0):
        #     if(back_img[tempx, tempy, 3] > tempz):
        #         back_img[tempx, tempy, 3] = tempz
        # elif (back_img[tempx, tempy, 3] != 0):
        #     if (back_img[tempx, tempy, 3] - tempz) < 0.05 and back_img[tempx, tempy, 3] > tempz:
        #         back_img[tempx, tempy, 3] = tempz

    for i in range(0, im_size):
        for j in range(0, im_size):
            # if back_img[i, j, 3] > img[i, j, 3]:
            #     back_img[i, j, 3] = img[i, j, 3]
            if (back_img[i, j, 3] == 0):
                # if(back_img[i,j,3] > img[i,j,3] and img[i,j,3] < -4):
                if back_img[i, j, 3] > img[i, j, 3]:
                    back_img[i, j, 3] = img[i, j, 3]
            if (back_img[i, j, 3] != 0):
                # if(back_img[i,j,3] - img[i,j,3]) < 0.05 and back_img[i,j,3] > img[i,j,3]:
                if back_img[i, j, 3] > img[i, j, 3]:
                    back_img[i, j, 3] = img[i, j, 3]
    return back_img


# 不用了
@jit(nopython=True)
def crop_back(img, PC_data, grid_size):
    grid_size_inv = 1 / grid_size
    im_size = round(200 * grid_size_inv)
    min_x, min_y = -100, -100
    PC_data = PC_data[PC_data[:, 2] > -5.5]
    point_num = PC_data.shape[0]
    in_points = np.zeros((point_num, 4))
    count_in = 0
    if point_num == 0:
        print('there is no data')
    for i in range(point_num):
        temppoint = PC_data[i]
        tempx = math.floor((PC_data[i, 0] - min_x) * grid_size_inv)
        tempy = math.floor((PC_data[i, 1] - min_y) * grid_size_inv)
        t_z = PC_data[i, 2]
        if (tempx > im_size - 1) or (tempx < 0) or (tempy > im_size - 1) or (tempy < 0):
            continue
        elif (abs(t_z - img[tempx, tempy, 3]) > 0.2):
            in_points[count_in, :] = temppoint
            count_in += 1

    in_points = in_points[:count_in, :]
    return in_points


@jit(nopython=True)
def get_bag_data(packet_data, ver_hori_angle):
    ###读一包数据
    bagdata = np.zeros((320, 4))
    for i in range(0, 1320, 132):
        h_temp = int((packet_data[2 + i] + packet_data[3 + i] * 256) / 10)  ###+352（+1630）,-270(y为负)

        for j in range(32):
            index = 32 * (i // 132) + j
            intesty = packet_data[6 + i + j * 4] / 255
            dis_temp = (packet_data[4 + i + j * 4] + packet_data[5 + i + j * 4] * 256) * 0.004
            bagdata[index, 3] = intesty
            h_temp_index = h_temp % 3600
            if j % 2 == 0 and j < 23:
                h_temp_index = (40 + h_temp) % 3600

            bagdata[index, 0] = dis_temp * ver_hori_angle[j, h_temp_index - 1, 0]
            bagdata[index, 1] = -dis_temp * ver_hori_angle[j, h_temp_index - 1, 1]
            bagdata[index, 2] = dis_temp * ver_hori_angle[j, h_temp_index - 1, 2]

    now_angle = (h_temp / 10) % 360
    head_data = np.ones(30)
    # print(head_data)
    for j in range(1320, len(packet_data)):
        head_data[j - 1320] = packet_data[j]
    # print(head_data)
    # print(f'head_data is {head_data[16:24]}')

    typett = type(bagdata[0][0])
    typett2 = type(now_angle)
    typett3 = type(head_data[0])
    # print("get_bag_data:", typett, typett2, typett3)

    return (bagdata, now_angle, head_data)
    # return (bagdata, now_angle)


def init_angle():
    ######角度分布#############
    deg = 0
    ver_hori_angle = np.zeros((32, 3600, 3))
    for m in range(32):
        if m == 0:
            deg = 0
        elif m <= 15:
            deg -= 0.6
        elif m <= 23:
            deg -= 1
        elif m <= 27:
            deg -= 2
        else:
            deg -= 3

        vert_cos = math.cos(PI_rad * deg)
        vert_sin = math.sin(PI_rad * deg)
        for n in range(3600):
            ver_hori_angle[m, n, 0] = vert_cos * math.sin(PI_rad * n / 10)  ##x
            ver_hori_angle[m, n, 1] = vert_cos * math.cos(PI_rad * n / 10)  ##y
            ver_hori_angle[m, n, 2] = vert_sin  ##z

    return ver_hori_angle


def read_pcap_offline(filename):
    #########解析数据############
    # filename = r'./2.pcap'
    last_angle = -1
    with open(filename, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        for timestamp, raw_buf in pcap:  # timestamp时间戳，raw_buf包中的原始数据
            eth = dpkt.ethernet.Ethernet(raw_buf)
            # 判断这个包是不是IP数据报
            if not isinstance(eth.data, dpkt.ip.IP):
                print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
                continue

            if not isinstance(eth.data.data, dpkt.udp.UDP):  # 解包，判断传输层协议是否是TCP，即当你只需要TCP时，可用来过滤
                print('it is not UDP')
                continue
            packet = eth.data  # 让以太网帧的ip数据报部分等于一个新的对象，packet
            if list(packet.dst) == [192, 168, 2, 88] and list(packet.src) == [192, 168, 2, 86]:

                udp_data = packet.data  ##每个UDP数据
                pl_data = udp_data.data  ##每一包的数据包含的东西
                packet_data = np.array([int(i) for i in pl_data])
                bag_data, now_angle = get_bag_data(packet_data)
                if now_angle > last_angle:
                    if last_angle == -1:
                        PC_data = bag_data
                        last_angle = now_angle
                    else:

                        PC_data = np.concatenate((PC_data, bag_data), axis=0)
                        last_angle = now_angle
                else:
                    last_angle = 0
                    PC_data = bag_data


# 过滤检测结果
@jit(nopython=True)
def filter_label_forKITTI(labels, scores, mode):
    # taked_indics = []
    # for idx in range(num_labels):
    #     if labels[idx] == 0 and scores[idx] > 0.2:
    #         taked_indics.append(idx)
    #     elif labels[idx] == 1 and scores[idx] >= 0.025:
    #         taked_indics.append(idx)
    #     elif labels[idx] == 2 and scores[idx] >= 0.025:
    #         taked_indics.append(idx)

    taked_indics = np.ones_like(labels)
    count_idx = 0
    num_labels = len(labels)
    if mode == 0:
        ##########针对不同种类使用不同的概率阈值选取合适的box_lidar#############
        for idx in range(num_labels):
            if labels[idx] == 0 and scores[idx] > 0.1:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 1 and scores[idx] >= 0.05:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 2 and scores[idx] >= 0.05:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 3 and scores[idx] >= 0.05:
                taked_indics[count_idx] = idx
                count_idx += 1
        ############################end##################
    elif mode == 1:
        #############针对不同种类box中的点数进行过滤#########
        for idx in range(num_labels):
            if labels[idx] == 0:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 1:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 2:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 3:
                taked_indics[count_idx] = idx
                count_idx == 1
        ###################end**end#####################
    else:
        ###################过滤不同种类符合概率和box内点数#########
        for idx in range(num_labels):
            if labels[idx] == 0 and (scores[idx]) > (0.01):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 1 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 2 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 3 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx == 1

    ###################只对行人和自行车进行过滤##########
    # for idx in range(num_labels):
    #
    #     if labels[idx] == 1 and scores[idx] >= 0.025:
    #         taked_indics[count_idx] = idx
    #         count_idx += 1
    #     elif labels[idx] == 2 and scores[idx] > 0.015:
    #         taked_indics[count_idx] = idx
    #         count_idx += 1
    #######################end###########################
    return taked_indics[:count_idx]


@jit(nopython=True)
def filter_label_forNuScenes(labels, scores, mode, fCarThreshold=0.25, fConeThreshold=0.9):
    taked_indics = np.ones_like(labels)
    count_idx = 0
    num_labels = len(labels)
    ##0:car,1:bike,2:bus,3:motorbike,4:pedestrian,5:traffic cone
    if mode == 0:
        ##########针对不同种类使用不同的概率阈值选取合适的box_lidar#############
        for idx in range(num_labels):
            # if labels[idx] == 0 and scores[idx] > 0.25:
            if labels[idx] == 0 and scores[idx] > fCarThreshold:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 1 and scores[idx] >= 0.02:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 2 and scores[idx] >= 0.3:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 3 and scores[idx] >= 0.02:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 4 and scores[idx] >= 0.02:
                taked_indics[count_idx] = idx
                count_idx += 1
            # elif labels[idx] == 5 and scores[idx] >= 0.9:
            elif labels[idx] == 5 and scores[idx] >= fConeThreshold:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 6 and scores[idx] >= 0.2:
                taked_indics[count_idx] = idx
                count_idx += 1
        ############################end##################
    elif mode == 1:
        #############针对不同种类box中的点数进行过滤#########
        for idx in range(num_labels):
            if labels[idx] == 0:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 1:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 2:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 3:
                taked_indics[count_idx] = idx
                count_idx == 1
            elif labels[idx] == 4:
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 5:
                taked_indics[count_idx] = idx
                count_idx == 1
            elif labels[idx] == 6:
                taked_indics[count_idx] = idx
                count_idx == 1
        ###################end**end#####################
    else:
        ###################过滤不同种类符合概率和box内点数#########
        for idx in range(num_labels):
            if labels[idx] == 0 and (scores[idx]) > (0.01):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 1 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 2 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 3 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx == 1
            elif labels[idx] == 4 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx += 1
            elif labels[idx] == 5 and (scores[idx]) > (0.05):
                taked_indics[count_idx] = idx
                count_idx == 1

    return taked_indics[:count_idx]


# 不用了
def simple_vis(boxes_corners, boxes_corners_trackers, points):
    linesets = []
    num_boxes = boxes_corners.shape[0]
    num_trackers_boxes = boxes_corners_trackers.shape[0]
    ALL_box_num = num_boxes + num_trackers_boxes
    print(f'All_box_num is {ALL_box_num}')
    for i in range(ALL_box_num):
        if i < num_boxes:
            points_box = boxes_corners[i]
            lines_box = np.array(
                [[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_box)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            linesets.append(line_set)
        else:
            points_box = boxes_corners_trackers[i - num_boxes]
            lines_box = np.array(
                [[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_box)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            linesets.append(line_set)

    points1 = points[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.paint_uniform_color([0, 0, 0])
    # point_cloud.points.size=0.1
    point_cloud.points = o3d.utility.Vector3dVector(points1)
    linesets.append(point_cloud)
    o3d.visualization.draw_geometries(linesets)


# 背景绿波
@jit(nopython=True)
def fused_back(backdata, grid_size_small, grid_size_big):
    grid_small_inv = 1 / grid_size_small
    im_size_small = round(200 * grid_small_inv)     ## range ： 200m

    grid_big_inv = 1 / grid_size_big
    im_size_big = round(200 * grid_big_inv)

    min_x, min_y = -100, -100

    img_small = np.zeros((im_size_small, im_size_small, 4))

    img_big = np.zeros((im_size_big, im_size_big, 4))

    # backdata = backdata[backdata[:, 2]>-5.5]
    point_num = backdata.shape[0]
    if point_num == 0:
        print('there is no data')
    for i in range(point_num):
        # if i == 108:
        #     print(" ")
        tempx_small = math.floor((backdata[i, 0] - min_x) * grid_small_inv)
        tempy_small = math.floor((backdata[i, 1] - min_y) * grid_small_inv)

        tempx_big = math.floor((backdata[i, 0] - min_x) * grid_big_inv)
        tempy_big = math.floor((backdata[i, 1] - min_y) * grid_big_inv)

        tempz = backdata[i, 2]

        if (tempx_small > im_size_small - 1) or (tempx_small < 0) or (tempy_small > im_size_small - 1) or (
                tempy_small < 0):
            continue
        elif (tempz < img_small[tempx_small, tempy_small, 3]):
            img_small[tempx_small, tempy_small, 3] = tempz

        elif (tempz < img_big[tempx_big, tempy_big, 3]):
            img_big[tempx_big, tempy_big, 3] = tempz
    # img_small = np.array(img_small, dtype=np.float32)
    # img_big = np.array(img_big,dtype=np.float32)
    # np.save("/data/pre_process/fuse_test/img_small/" + str(kk) + ".npy", img_small)
    # np.save("/data/pre_process/fuse_test/img_big/" + str(kk) + ".npy", img_big)
    # img_small_cpp = np.load("/data/pre_process/pre_process/cmake-build-debug/img_small.npy")
    # img_big_cpp = np.load("/data/pre_process/pre_process/cmake-build-debug/img_big.npy")

    return img_small, img_big


# 过滤　与fused_back配合使用
@jit(nopython=True)
def fused_crop_back(img_small, img_big, PC_data, grid_size_small, grid_size_big, thresh_small, thresh_big, h_thresh_x,
                    h_thresh_y, dis_thresh):
    grid_small_inv = 1 / grid_size_small
    im_size_small = round(200 * grid_small_inv)

    grid_big_inv = 1 / grid_size_big

    min_x, min_y = -100, -100
    point_num = PC_data.shape[0]
    in_points = np.zeros((point_num, 4))
    out_points = np.zeros((point_num, 4))
    count_in = 0
    count_out = 0
    if point_num == 0:
        print('there is no data')

    for i in range(point_num):
        temppoint = PC_data[i]
        tempx_small = math.floor((PC_data[i, 0] - min_x) * grid_small_inv)
        tempy_small = math.floor((PC_data[i, 1] - min_y) * grid_small_inv)

        tempx_big = math.floor((PC_data[i, 0] - min_x) * grid_big_inv)
        tempy_big = math.floor((PC_data[i, 1] - min_y) * grid_big_inv)

        if (tempx_small > im_size_small - 1) or (tempx_small < 0) or (tempy_small > im_size_small - 1) or (
                tempy_small < 0):
            continue
        elif (PC_data[i][2] - img_small[tempx_small, tempy_small, 3]) > thresh_small:
            in_points[count_in, :] = temppoint
            count_in += 1
        # elif abs(PC_data[i][0]) > 50 or abs(PC_data[i][1]) > 50:
        #     if ((PC_data[i][2] - img_big[tempx_big, tempy_big, 3]) > h_thresh):
        #         in_points[count_in, :] = temppoint
        #         count_in += 1
        #     else:
        #         out_points[count_out, :] = temppoint
        #         count_out += 1
        elif abs(PC_data[i][0]) < dis_thresh and abs(PC_data[i][1]) < dis_thresh:
            if ((PC_data[i][2] - img_big[tempx_big, tempy_big, 3]) > thresh_big):
                in_points[count_in, :] = temppoint
                count_in += 1
            else:
                out_points[count_out, :] = temppoint
                count_out += 1
        elif (abs(PC_data[i][0]) > dis_thresh):
            if PC_data[i][2] > h_thresh_x:
                in_points[count_in, :] = temppoint
                count_in += 1
            else:
                out_points[count_out, :] = temppoint
                count_out += 1
        elif (abs(PC_data[i][1]) > dis_thresh):
            if PC_data[i][2] > h_thresh_y:
                in_points[count_in, :] = temppoint
                count_in += 1
            else:
                out_points[count_out, :] = temppoint
                count_out += 1
        else:
            out_points[count_out, :] = temppoint
            count_out += 1
    in_points = in_points[:count_in, :]
    out_points = out_points[:count_out, :]
    # print('filter number: ')
    # print(count_in)
    # in_points = np.array(in_points, dtype=np.float32)
    # out_points = np.array(out_points, dtype=np.float32)

    return in_points, out_points


def read_bmp_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cache_bmp(img)
    return img


def filter_points(points, img_path, crop = True, filter_ground = False):
    img = read_bmp_img(img_path)
    if crop and filter_ground:
        res = np.array(crop_pcdata(points, img), dtype=np.float32)
        img_small, img_big = fused_back(res, 1, 5)
        in_point, out_point = fused_crop_back(img_small, img_big, res, 1, 5, 0.2, 0.4, -7, -7, 80)
        return in_point
    elif crop and not filter_ground:
        return np.array(crop_pcdata(points, img), dtype=np.float32)
    elif not crop and filter_ground:
        img_small, img_big = fused_back(points, 1, 5)
        in_point, out_point = fused_crop_back(img_small, img_big, res, 1, 5, 0.2, 0.4, -7, -7, 80)
        return in_point
    else:
        return points

class FilterPoints:
    def __init__(self, bmp_path):
        self.bmp = None
        if bmp_path is not None:
            self.bmp = read_bmp_img(bmp_path)

    @staticmethod
    def read_bmp_img(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cache_bmp(img)
        return img

    def filter_points(self, points, crop=True, filter_ground=False):
        if self.bmp is None:
            return points
        if crop and filter_ground:
            res = np.array(crop_pcdata(points, self.bmp), dtype=np.float32)
            img_small, img_big = fused_back(res, 1, 5)
            in_point, out_point = fused_crop_back(img_small, img_big, res, 1, 5, 0.2, 0.4, -7, -7, 80)
            return in_point
        elif crop and not filter_ground:
            return np.array(crop_pcdata(points, self.bmp), dtype=np.float32)
        elif not crop and filter_ground:
            img_small, img_big = fused_back(points, 1, 5)
            in_point, out_point = fused_crop_back(img_small, img_big, res, 1, 5, 0.2, 0.4, -7, -7, 80)
            return in_point
        else:
            return points

if __name__ == "__main__":

    # img = cv2.imread("xiaopi2.png")
    # cv2.imwrite("xiaopi2.bmp", img)

    pc_data = np.load("/home/cxy/Downloads/12344.npy")
    custom_draw_geometry(pc_data)
    # save_bmp(pc_data, 2000, "xiaopi.png")


    """
    # save_bmp(pc_data, 2000, "/data/pre_process/bmp.png")
    img = read_img("/home/cxy/Documents/second_kd/second/pytorch/utils/new_crop.bmp")

    data_path = "/home/cxy/Documents/OpenPCDet/data/double_lidar_all_data/training/bin/"
    pc_files = os.listdir(data_path)

    for i in range(len(pc_files)):
        pc_data = np.fromfile(data_path + pc_files[i], dtype=np.float32, count=-1).reshape((-1, 4))
        # save_bmp(pc_data, 2000, "./new_crop.bmp")

        unground_points = get_unground_points(pc_data, img)
        custom_draw_geometry(unground_points)
    """