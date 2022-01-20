import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch
import time
import torchplus
from torch.nn import functional as F
from google.protobuf import text_format
from second.protos import pipeline_pb2
from second.core import box_np_ops
from second.utils import config_tool
from second.core.inference import InferenceContext
from second.builder import target_assigner_builder, voxel_builder
from second.pytorch.builder import box_coder_builder, second_builder
from second.pytorch.models.voxelnet import VoxelNet
from second.pytorch.train import example_convert_to_torch
from second.pytorch.utils.open3d_utils import *
from second.pytorch.train import build_network
from second.pytorch.utils.pre_process import FilterPoints

__pctype__ = {
    "bin": read_bin,
    "pcd": read_pcd,
    "npy": read_npy
}

def resize_weight(weight):
    sizes = list(weight.cpu().numpy().shape)
    sizes = [sizes[-1]] + sizes[:-1]
    
    weight = weight.reshape(sizes)
    
    return weight
def read_gt(label_path):
    bboxs = []
    for line in open(label_path):
        line = line.strip().split(",")
        bbox = [float(line[2]) / 100,
                float(line[3]) / 100,
                float(line[4]) / 100,
                float(line[8]) / 100,
                float(line[7]) / 100,
                float(line[9]) / 100,
                math.pi / 2 * 3 - float(line[6]) * math.pi / 180.0
                ]
        bboxs.append(bbox)
    bboxs = np.array(bboxs)
    return bboxs


def filter_conf(dets, conf):
    index = len(dets["scores"])
    for i in range(len(dets["scores"])):
        if dets["scores"][i] < conf:
            index = i
            break
    dets["box3d_lidar"] = dets["box3d_lidar"][:index]
    dets["scores"] = dets["scores"][:index]
    dets["label_preds"] = dets["label_preds"][:index]
    return dets


def inference_video(args):
    if args.show:
        # ===================== open3d windows ==============================
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Wanji 32line Lidar', width=1080, height=720, left=300, top=150, visible=True)
        point_cloud = o3d.geometry.PointCloud()
        line_sets = [o3d.geometry.LineSet() for _ in range(250)]
        vis.add_geometry(point_cloud)

        render_option = vis.get_render_option()
        render_option.point_size = 1
        render_option.background_color = np.asarray([0, 0, 0])
        to_reset_view_point = True

        line_range = get_line_range(80)
        line_sets.append(line_range)
        for line in line_sets:
            vis.add_geometry(line)
        grips = grid_map(200)
        for i in grips:
            vis.add_geometry(i)
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
        # ==============================================================================================

    files = []
    if os.path.isdir(args.pointcloud):
        files = os.listdir(args.pointcloud)
        files = [os.path.join(args.pointcloud, i) for i in files]
    else:
        files.append(args.pointcloud)
    files.sort()

    gt_files = []
    if args.gt_label is not None:
        if os.path.exists(args.gt_label) and os.path.isdir(args.gt_label):
            gt_files = os.listdir(args.gt_label)
            gt_files = [os.path.join(args.gt_label, i) for i in gt_files]
        else:
            gt_files.append(args.gt_label)
        gt_files.sort()
    point_cloud_type = "bin"
    try:
        point_cloud_type = files[0].split(".")[-1]
    except:
        print("Error! : The pointcloud dir has no file!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(args.config, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second

    net = build_network(model_cfg, measure_time=True, knowledge_distillation=not args.kd).to(device)

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)
    filter = FilterPoints(args.bmp)
    time_log = []
    for i in range(len(files)):
        points = __pctype__[point_cloud_type](os.path.join(args.pointcloud, files[i]))
        points = filter.filter_points(points, crop=True, filter_ground=False)

        t1 = time.time()
        res = voxel_generator.generate(points)
        # add batch idx to coords
        coords = F.pad(res, (1, 0), mode='constant', value=0).cuda()
        voxels = torch.tensor(points, dtype=torch.float32, device=device)
        example = {
            "anchors": anchors,
            "voxels": voxels,
            "coordinates": coords
        }
        pred = net(example)[0]

        t2 = time.time()
        pred_time = round((t2 - t1) * 1000, 3)
        print(files[i].split("/")[-1] + "; points: {}; detect {} boxes; inference cost : {} ms!".format(points.shape[0], pred["box3d_lidar"].shape[0], pred_time))
        time_log.append(pred_time)

        if args.save_dir is not None:
            bboxes = torch.cat(
                [pred["box3d_lidar"], pred["label_preds"].reshape((-1, 1)), pred["scores"].reshape((-1, 1))],
                1).cpu().numpy()
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            np.savetxt(os.path.join(args.save_dir, files[i].split("/")[-1].replace("bin", "csv")), bboxes,
                       delimiter=",")

        pred = filter_conf(pred, args.score_thr)

        if args.show:
            pred["box3d_lidar"] = pred["box3d_lidar"].cpu().numpy()
            box3d = box_np_ops.center_to_corner_box3d(pred["box3d_lidar"][:, :3],
                                                      pred["box3d_lidar"][:, 3:6],
                                                      pred["box3d_lidar"][:, 6],
                                                      origin=(0.5, 0.5, 0.5))

            for j in range(len(box3d)):
                points_box = box3d[j]
                lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                      [0, 4], [1, 5], [2, 6], [3, 7]])
                colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
                line_sets[j].points = o3d.utility.Vector3dVector(points_box)
                line_sets[j].lines = o3d.utility.Vector2iVector(lines_box)
                line_sets[j].colors = o3d.utility.Vector3dVector(colors)
            line_set_index = len(box3d)
            if len(gt_files) == 0:
                pass
            # if gt_files[i].split("/")[-1].split(".")[0] != files[i].split("/")[-1].split(".")[0]:
            #     print(">>>>>>>>>>> Error message: The pointcloud index and label index doesn't match!")
            else:
                gt_box = read_gt(gt_files[i])
                gt_box3d = box_np_ops.center_to_corner_box3d(gt_box[:, :3], gt_box[:, 3:6], gt_box[:, 6],
                                                             origin=(0.5, 0.5, 0.5))

                for j in range(len(gt_box3d)):
                    points_box = gt_box3d[j]
                    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                          [0, 4], [1, 5], [2, 6], [3, 7]])
                    colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
                    line_sets[j + line_set_index].points = o3d.utility.Vector3dVector(points_box)
                    line_sets[j + line_set_index].lines = o3d.utility.Vector2iVector(lines_box)
                    line_sets[j + line_set_index].colors = o3d.utility.Vector3dVector(colors)
                line_set_index += len(gt_box3d)

            for j in range(line_set_index, 250):
                points_box = np.zeros((8, 3))
                lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                      [0, 4], [1, 5], [2, 6], [3, 7]])
                colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
                line_sets[j].points = o3d.utility.Vector3dVector(points_box)
                line_sets[j].lines = o3d.utility.Vector2iVector(lines_box)
                line_sets[j].colors = o3d.utility.Vector3dVector(colors)

            if args.show:
                point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
                vis.update_geometry(point_cloud)
                for line in line_sets:
                    vis.update_geometry(line)
                if to_reset_view_point:
                    vis.reset_view_point(True)
                    to_reset_view_point = False
                vis.poll_events()
                vis.update_renderer()
        
    print("================= Done! =================")
    
    print(">>> average inference time : {} ms!".format(round(np.mean(time_log[1:]), 3)))

def inference_frame(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(args.config, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    net = build_network(model_cfg, measure_time=True, knowledge_distillation=not args.kd).to(device)

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    net.load_state_dict(torch.load(args.checkpoint))

    net.eval()
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)
    filter = FilterPoints(args.bmp)

    files = []
    if os.path.isdir(args.pointcloud):
        files = os.listdir(args.pointcloud)
        files = [os.path.join(args.pointcloud, i) for i in files]
    else:
        files.append(args.pointcloud)
    files.sort()

    gt_files = []
    if args.gt_label is not None:
        if os.path.isdir(args.gt_label):
            gt_files = os.listdir(args.gt_label)
            gt_files = [os.path.join(args.gt_label, i) for i in gt_files]
        else:
            gt_files.append(args.gt_label)
        gt_files.sort()
    time_log = []
    for i in range(len(files)):
        points = __pctype__[files[i].split(".")[-1]](files[i])
        points = filter.filter_points(points, crop=True, filter_ground=False)
        t1 = time.time()
        res = voxel_generator.generate(points)
        # add batch idx to coords
        coords = F.pad(res, (1, 0), mode='constant', value=0).cuda()
        voxels = torch.tensor(points, dtype=torch.float32, device=device)
        example = {
            "anchors": anchors,
            "voxels": voxels,
            "coordinates": coords
        }
        pred = net(example)[0]

        t2 = time.time()
        pred_time = round((t2 - t1) * 1000, 3)
        time_log.append(pred_time)
        print(files[i].split("/")[-1] + "; points: {}; detect {} boxes; inference cost : {} ms!".format(points.shape[0], pred["box3d_lidar"].shape[0], pred_time))
        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)

            bboxes = torch.cat(
                [pred["box3d_lidar"], pred["label_preds"].reshape((-1, 1)), pred["scores"].reshape((-1, 1))],
                1).cpu().numpy()
            save_path = os.path.join(args.save_dir, files[i].split("/")[-1].split(".")[0] + ".csv")
            np.savetxt(save_path, bboxes, delimiter=",")
        pred = filter_conf(pred, args.score_thr)
        if args.show:
            pred["box3d_lidar"] = pred["box3d_lidar"].cpu().numpy()
            box3d = box_np_ops.center_to_corner_box3d(pred["box3d_lidar"][:, :3],
                                                      pred["box3d_lidar"][:, 3:6],
                                                      pred["box3d_lidar"][:, 6],
                                                      origin=(0.5, 0.5, 0.5))
            line_sets = [o3d.geometry.LineSet() for _ in range(len(box3d))]
            for j in range(len(box3d)):
                points_box = box3d[j]
                lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                      [0, 4], [1, 5], [2, 6], [3, 7]])
                colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
                line_sets[j].points = o3d.utility.Vector3dVector(points_box)
                line_sets[j].lines = o3d.utility.Vector2iVector(lines_box)
                line_sets[j].colors = o3d.utility.Vector3dVector(colors)
            if len(gt_files) == 0:
                pass
            # if gt_files[i].split("/")[-1].split(".")[0] != files[i].split("/")[-1].split(".")[0]:
            #     print(">>>>>>>>>>> Error message: The pointcloud index and label index doesn't match!")
            else:
                gt_box = read_gt(gt_files[i])
                gt_box3d = box_np_ops.center_to_corner_box3d(gt_box[:, :3], gt_box[:, 3:6], gt_box[:, 6],
                                                             origin=(0.5, 0.5, 0.5))
                line_sets_gt = [o3d.geometry.LineSet() for _ in range(len(gt_box3d))]
                for j in range(len(gt_box3d)):
                    points_box = gt_box3d[j]
                    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                          [0, 4], [1, 5], [2, 6], [3, 7]])
                    colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
                    line_sets_gt[j].points = o3d.utility.Vector3dVector(points_box)
                    line_sets_gt[j].lines = o3d.utility.Vector2iVector(lines_box)
                    line_sets_gt[j].colors = o3d.utility.Vector3dVector(colors)
                line_sets.extend(line_sets_gt)
                line_sets_gt.clear()
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
            line_range = get_line_range()
            line_sets.append(line_range)
            custom_draw_geometry(point_cloud, line_sets)
    print("================= Done! =================")
    print(">>> Average inference time : {} ms!".format(round(np.mean(time_log[1:]),3)))



def main():
    parser = ArgumentParser()
    parser.add_argument('--pointcloud',
                        default="/data/Documents/second_kd/datasets/gate8_double/test/bin",
                        help='pointcloud file or dir')
    parser.add_argument('--config', default="/data/Documents/second_dv/all.wjdata.semi.config", help='config file')
    parser.add_argument('--checkpoint', default="/data/Documents/second_dv/second/trainmodel/dv/voxelnet-32250.tckpt",
                        help='Checkpoint file')
    parser.add_argument('--bmp', default="/data/Documents/second_dv/crop_gate8.bmp", help='bmp_img"/data/second_dv/crop_gate8.bmp"')
    parser.add_argument('--score_thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='dir to save results')
    parser.add_argument('--show', default=True, help='show online visualization results')
    parser.add_argument('--gt_label', default=None,
                        help='show one by one')
    parser.add_argument('--kd', default=True, help='Knowledge distillation')
    args = parser.parse_args()

    # inference on single frame data
    # inference_frame(args)

    # inference on pointcloud flow data
    inference_video(args)


if __name__ == "__main__":
    main()
