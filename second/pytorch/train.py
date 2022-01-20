# coding=gbk
import open3d as o3d
import copy
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import time
import re 
import numpy as np
import torch
from second.core import box_np_ops
from google.protobuf import text_format
from second.utils import config_tool
import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.KD_LOSS.kd_loss import compute_lost_KD, compute_lost_KD2
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
from second.pytorch.utils.pre_process import *
import psutil
import pdb
import os
from utils.sort import *
from prettytable import PrettyTable
from second.data.wjlidar_common import pcd_to_npy
from second.pytorch.utils.open3d_utils import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

center = np.array([[0, 0, -4.5, 140.8, 140.8, 0.1, 0]])
range_coor = box_np_ops.center_to_corner_box3d(
    center[:, :3],
    center[:, 3:6],
    center[:, 6])[0]
line_range = o3d.geometry.LineSet()
line_range.points = o3d.utility.Vector3dVector(range_coor)
line_range.lines = o3d.utility.Vector2iVector(
    np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
              [0, 4], [1, 5], [2, 6], [3, 7]]))
line_range.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for j in range(12)]))


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in  ["coordinates"]:
            example_torch[k] = v.cuda()
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


"""
knowledge_distillation=False:  student network
knowledge_distillation=True :  teacher network
moren is student
"""
def build_network(model_cfg, measure_time=False, knowledge_distillation=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim

    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time, knowledge_distillation=knowledge_distillation)
    return net

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def freeze_params(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue 
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue 
        remain_params.append(p)
    return remain_params

def freeze_params_v2(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False

def filter_param_dict(state_dict: dict, include: str=None, exclude: str=None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue 
        res_dict[k] = p
    return res_dict


def train(config_path,
          model_dir,
          result_path=None,
          create_folder=False,
          display_step=1,
          summary_step=5,
          pretrained_path=None,
          pretrained_include=None,
          pretrained_exclude=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=True,
          ckpt_path=None,
          kd=False):
    """train a VoxelNet model specified by a config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    
    net = build_network(model_cfg, measure_time, knowledge_distillation=not kd).to(device)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    #print("num parameters:", len(list(net.parameters())))
    #torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v        
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items():
            print(k, v.shape)
        model_dict.update(new_pretrained_dict) 
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)
        net.clear_global_step()
        net.clear_metrics()
    if multi_gpu:
        net_parallel = torch.nn.DataParallel(net)
    else:
        net_parallel = net
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=False,
        loss_scale=loss_scale)
    if loss_scale < 0:
        loss_scale = "dynamic"
    if train_cfg.enable_mixed_precision:
        max_num_voxels = input_cfg.preprocess.max_number_of_voxels * input_cfg.batch_size
        assert max_num_voxels < 65535, "spconv fp16 training only support this"
        from apex import amp
        net, amp_optimizer = amp.initialize(net, fastai_optimizer,
                                        opt_level="O2",
                                        keep_batchnorm_fp32=True,
                                        loss_scale=loss_scale
                                        )
        net.metrics_to_float()
    else:
        amp_optimizer = fastai_optimizer
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    if multi_gpu:
        num_gpu = torch.cuda.device_count()
        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_second_batch
        num_gpu = 1
    # =========== KD ============
    # teacher model
    if kd != 0:
        evaluate_net = build_network(model_cfg, measure_time=measure_time,knowledge_distillation=True).to(device)
        torchplus.train.restore(ckpt_path, evaluate_net)
        evaluate_net.eval()

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,
        shuffle=True,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not multi_gpu)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=True,
        collate_fn=merge_second_batch)

    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    try:
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in dataloader:
                lr_scheduler.step(net.get_global_step())
                time_metrics = example["metrics"]
                example.pop("metrics")
                example_torch = example_convert_to_torch(example, float_dtype)

                batch_size = example["anchors"].shape[0]

                # prev = example['metadata'][0]['prev']
                # if prev is '':
                #     example_torch.update({"box_pred": None})
                # else:
                #     example_torch.update({"box_pred": box_preds})

                # idx = example['metadata'][0]['idx']
                # if idx < 1:
                #     example_torch.update({"spatial_features": None})
                # else:
                #     example_torch.update({"spatial_features": spatial_features})

                ret_dict, student_dict = net_parallel(example_torch)
                # spatial_features = ret_dict['spatial_features']
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                #iou_loss = ret_dict["iou_loss"]

                ###frame loss###
                # frame_loss = ret_dict["frame_loss"]
                # box_preds = ret_dict["box_preds"]
                
                cared = ret_dict["cared"]
                labels = example_torch["labels"]
        
                if kd:
                    teacher_dict = evaluate_net(example_torch,knowledge_distillation=True)
                    soft_loss = compute_lost_KD2(example_torch, student_dict, teacher_dict)
                    loss += soft_loss

        
                if train_cfg.enable_mixed_precision:
                    with amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()
                amp_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics( cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step()

                if global_step % display_step == 0:
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0])
                    step_times = []
                    metrics.update(net_metrics)
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(amp_optimizer.lr),
                        "mem_usage": psutil.virtual_memory().percent,
                    }
                    model_logging.log_metrics(metrics, global_step)

                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    #pdb.set_trace()
                    net.eval()
                    result_path_step = result_path / f"step_{net.get_global_step()}"
                    result_path_step.mkdir(parents=True, exist_ok=True)
                    #model_logging.log_text("#################################",
                    #                    global_step)
                    #model_logging.log_text("# EVAL", global_step)
                    #model_logging.log_text("#################################",
                    #                    global_step)
                    #model_logging.log_text("Generate output labels...", global_step)
                    t = time.time()
                    detections = []
                    prog_bar = ProgressBar()
                    net.clear_timer()
                    prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                                // eval_input_cfg.batch_size)
                    for example in iter(eval_dataloader):
                        example = example_convert_to_torch(example, float_dtype)
                        detections += net(example)
                        prog_bar.print_bar()

                    sec_per_ex = len(eval_dataset) / (time.time() - t)
                    #model_logging.log_text(
                    #    f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                    #    global_step)
                    result_dict = eval_dataset.dataset.evaluation(
                        detections, str(result_path_step))
                    # print(result_dict)
                    # ==============================================================
                    print("\n")
                    table_res = PrettyTable(["class", "ap_50", "ap_25"])
                    for k, v in result_dict["detail"].items():
                        class_ = k
                        ap_50 = str(round(v['3d@0.50'][0], 2))
                        ap_25 = str(round(v['3d@0.25'][0], 2))
                        table_res.add_row([class_, ap_50, ap_25])
                    print(table_res)
                    print("\n")
                    # ==============================================================
                    with open(result_path_step / "result.pkl", 'wb') as f:
                        pickle.dump(detections, f)
                    net.train()
                step += 1
                if step >= total_step:
                    break
            if step >= total_step:
                break
    except Exception as e:
        print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                    step)
        raise e
    finally:
       model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                net.get_global_step())


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             measure_time=False,
             batch_size=None,
             **kwargs):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file and convert_detection_to_kitti_annos
    in second.data.kitti_dataset.
    """
    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=measure_time, knowledge_distillation=False).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    # pdb.set_trace()
    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=True,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    detections = []
    # previous = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    all_time = 0
    for example in iter(eval_dataloader):
        # prev = example['metadata'][0]['prev']
        # if len(prev) > 0:
        #     gt_boxes0 = previous["pred_boxes"]
        #     # gt_boxes0 = np.expand_dims(gt_boxes0, axis=0)
        #     # example['anchors'] = np.concatenate((example['anchors'], gt_boxes0), axis=1)
        #     points0 = previous["points0"]
        #     gt_boxes0 = gt_boxes0.cpu().numpy()
        #     num_obj = gt_boxes0.shape[0]
        #     point_indices = box_np_ops.points_in_rbbox(points0, gt_boxes0)
        #     gt_points = []
        #     for i in range(num_obj):
        #         gt_points0 = points0[point_indices[:, i]]
        #         gt_points.append(gt_points0)
        #     gt_points = np.concatenate(gt_points, axis=0)
        #     res = voxel_generator.generate(gt_points)
        #     voxels = res["voxels"]
        #     coordinates = res["coordinates"]
        #     coors = np.zeros(coordinates.shape[0])
        #     coors = np.expand_dims(coors, axis=1)
        #     coordinate = np.append(coors, coordinates, axis=1)
        #     num_points = res["num_points_per_voxel"]
        #     num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        #     num_voxels = np.expand_dims(num_voxels, axis=1)
        #     example['voxels'] = np.concatenate((example['voxels'], voxels), axis=0)
        #     example['coordinates'] = np.concatenate((example['coordinates'], coordinate), axis=0)
        #     example['num_points'] = np.concatenate((example['num_points'], num_points), axis=0)
        #     example['num_voxels'] = np.sum((example['num_voxels'], num_voxels), axis=0)
        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        with torch.no_grad():
            t1 = time.time()
            detection = net(example)
            t2 = time.time()
            all_time += ((t2 - t1) * 1000)
            #print("================ cost_time {} ms!".format((t2 - t1)* 1000))
            detections += detection
            # points0 = np.squeeze(example['points'])
            # pred_boxes = detection[0]['box3d_lidar']
            # spatial_features = detection[0]
            # previous = {
            #     "points0": points0,
            #     "pred_boxes": pred_boxes,
            # }
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    #with open("result.pkl", 'wb') as f:
    #    pickle.dump(detections, f)
    #pdb.set_trace()
    result_dict = eval_dataset.dataset.evaluation(detections,
                                                  str(result_path_step))
    print(result_dict)
    #if result_dict is not None:
    #   for k, v in result_dict["detail"].items():
    #       print("Evaluation {}".format(k))
    #        print(v)

def helper_tune_target_assigner(config_path, target_rate=None, update_freq=200, update_delta=0.01, num_tune_epoch=5):
    """get information of target assign to tune thresholds in anchor generator.
    """    
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, False)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    
    class_count = {}
    anchor_count = {}
    class_count_tune = {}
    anchor_count_tune = {}
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
        class_count_tune[c] = 0
        anchor_count_tune[c] = 0


    step = 0
    classes = target_assigner.classes
    if target_rate is None:
        num_tune_epoch = 0
    for epoch in range(num_tune_epoch):
        for example in dataloader:
            gt_names = example["gt_names"]
            for name in gt_names:
                class_count_tune[name] += 1
            
            labels = example['labels']
            for i in range(1, len(classes) + 1):
                anchor_count_tune[classes[i - 1]] += int(np.sum(labels == i))
            if target_rate is not None:
                for name, rate in target_rate.items():
                    if class_count_tune[name] > update_freq:
                        # calc rate
                        current_rate = anchor_count_tune[name] / class_count_tune[name]
                        if current_rate > rate:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold += update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold += update_delta
                        else:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold -= update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold -= update_delta
                        anchor_count_tune[name] = 0
                        class_count_tune[name] = 0
            step += 1
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
    total_voxel_gene_time = 0
    count = 0

    for example in dataloader:
        gt_names = example["gt_names"]
        total_voxel_gene_time += example["metrics"][0]["voxel_gene_time"]
        count += 1

        for name in gt_names:
            class_count[name] += 1
        
        labels = example['labels']
        for i in range(1, len(classes) + 1):
            anchor_count[classes[i - 1]] += int(np.sum(labels == i))
    print("avg voxel gene time", total_voxel_gene_time / count)

    print(json.dumps(class_count, indent=2))
    print(json.dumps(anchor_count, indent=2))
    if target_rate is not None:
        for ag in target_assigner._anchor_generators:
            if ag.class_name in target_rate:
                print(ag.class_name, ag.match_threshold, ag.unmatch_threshold)

def mcnms_parameters_search(config_path,
          model_dir,
          preds_path):
    pass

def pc_video_demo(config_path,
          ckpt_path,
          kd = False,
          pc=None):
    """train a VoxelNet model specified by a config file.
        """

    # ========================== sort ========================
    # tracker = Sort()
    # predict_path = "/media/cxy/������-7����/20211020000000/res/"
    # # save_path = "/media/cxy/������-7����/20211020000000/1027/"
    # line_k = np.load("/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_k.npy")
    # line_b = np.load("/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_b.npy")
    # line_limit = np.load(
    #     "/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_limit.npy")
    # line_angle = np.load(
    #     "/home/cxy/Documents/second_kd/second/pytorch/utils/chedao/empty_chedao_param/chedao_angle.npy")
    # =====================================================================

    # ===================== open3d windows ==============================
    point_cloud = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Wanji 32line Lidar', width=1080, height=720, left=300, top=150, visible=True)
    vis.add_geometry(point_cloud)
    line_sets = [o3d.geometry.LineSet() for _ in range(200)]

    render_option = vis.get_render_option()
    render_option.point_size = 1
    render_option.background_color = np.asarray([0, 0, 0])  # ��ɫ 0Ϊ�ڣ�1Ϊ��
    to_reset_view_point = True

    center = np.array([[0, 0, -4.5, 140.8, 140.8, 0.1, 0]])
    range_coor = box_np_ops.center_to_corner_box3d(
        center[:, :3],
        center[:, 3:6],
        center[:, 6])[0]
    line_range = o3d.geometry.LineSet()
    line_range.points = o3d.utility.Vector3dVector(range_coor)
    line_range.lines = o3d.utility.Vector2iVector(
        np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                  [0, 4], [1, 5], [2, 6], [3, 7]]))
    line_range.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1] for j in range(12)]))
    line_sets.append(line_range)
    for line in line_sets:
        vis.add_geometry(line)
    grips = grid_map(200)
    for i in grips:
        vis.add_geometry(i)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # ==============================================================================================

    # data_path
    data_path = "/data/Documents/second_kd/zhangshichen/"
    save_path = "/data/Documents/recall_calculation/guanxijia_test/res2/"
    # img = read_img("/media/cxy/DATA/project_guanxijia/shijian/bmp/70/crop.bmp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second

    net = build_network(model_cfg, measure_time=True, knowledge_distillation=kd).to(device)

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    net.load_state_dict(torch.load(ckpt_path))

    net.eval()
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)
    t_all = 0
    pc_files = os.listdir(data_path)

    for i in range(384):
        # for i in range(len(pc_files)):
        # points = np.fromfile(data_path + str(i).zfill(4) + ".bin", dtype=np.float32, count=-1).reshape((-1, 4))
        points = read_pcd(data_path + str(i) + ".pcd")
        # points = np.load(data_path + str(i).zfill(4) + ".npy")
        points[:, 0] -= -25.15
        points[:, 1] -= 10



        # point_cloud = o3d.geometry.PointCloud()
        # points = get_unground_points(points, img)
        """voxels, coordinates, num_points_per_voxel, voxel_point_mask, voxel_num"""
        t1 = time.time()
        res = voxel_generator.generate(points, max_voxels=90000)
        t2 = time.time()
        voxels = res["voxels"]
        num_points = res["num_points_per_voxel"]
        coords = res["coordinates"]
        # add batch idx to coords
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
        example = {
            "anchors": anchors,
            "voxels": voxels,
            "num_points": num_points,
            "coordinates": coords,
        }
        t3 = time.time()
        pred = net(example)[0]
        t4 = time.time()


        a = torch.cat([pred["box3d_lidar"], pred["label_preds"].reshape((-1, 1)), pred["scores"].reshape((-1, 1))], 1).cpu().numpy()
        # a = a[:, [0, 1, 3, 4, 6, 2, 5, 7, 8]]
        # res = tracker.update(pre_bbox, line_k, line_b, line_limit, line_angle).astype(np.float32)
        # # x, y, w, l, yaw, z, h, cls, speed, id, conf, hit, 0, 0, 0, 0
        # res = res[:, [0, 1, 5, 2, 3, 6, 4]]
        # res[:, -1] *= (math.pi / 180)
        # res[:, -1] += (math.pi / 2)
        # res[:, -1] = a[:, 6].reshape(-1)[::-1]

        print(i)
        # np.savetxt(save_path + str(i) + ".csv_error", a, delimiter=",")
        # """
        pred["box3d_lidar"] = pred["box3d_lidar"].cpu().numpy()
        box3d = box_np_ops.center_to_corner_box3d(pred["box3d_lidar"][:, :3],
                                                  pred["box3d_lidar"][:, 3:6],
                                                  pred["box3d_lidar"][:, 6],
                                                  origin=(0.5, 0.5, 0.5))
        # line_sets = [o3d.geometry.LineSet() for _ in range(len(box3d))]
        for j in range(len(box3d)):
            points_box = box3d[j]
            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
            line_sets[j].points = o3d.utility.Vector3dVector(points_box)
            line_sets[j].lines = o3d.utility.Vector2iVector(lines_box)
            line_sets[j].colors = o3d.utility.Vector3dVector(colors)
        for j in range(len(box3d), 200):
            points_box = np.zeros((8, 3))
            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([[1, 1, 1] for j in range(len(lines_box))])
            line_sets[j].points = o3d.utility.Vector3dVector(points_box)
            line_sets[j].lines = o3d.utility.Vector2iVector(lines_box)
            line_sets[j].colors = o3d.utility.Vector3dVector(colors)

        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        vis.update_geometry(point_cloud)
        for line in line_sets:
            vis.update_geometry(line)
        if to_reset_view_point:
            vis.reset_view_point(True)
            to_reset_view_point = False
        vis.poll_events()
        vis.update_renderer()
        
        # time.sleep(0.1)
        # """

def val_for_view(config_path,
          ckpt_path,
          kd = False,
          pc=None):
    """
    train a VoxelNet model specified by a config file.
    """
    f_path = "/data/Documents/second_kd/haidian/20211020120700-1/test/bin/"
    save_path = "/data/infer_res/gate8_res_kd2/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
#    net = build_network(model_cfg, False).to(device)
    net = build_network(model_cfg, measure_time=True, knowledge_distillation=kd).to(device)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    net.load_state_dict(torch.load(ckpt_path))

    net.eval()
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)
    t_all = 0

    pc_files = os.listdir(f_path)
    # img = read_img("/data/Documents/second_kd/second/pytorch/utils/new_crop.bmp")

    for i in range(5200):
    # for i in range(len(pc_files)):
        # points = np.fromfile(f_path + pc_files[i], dtype=np.float32, count=-1).reshape((-1, 4))
        # points = get_unground_points(points, img)
        # points = np.load(f_path + str(i).zfill(6) + ".npy")
        points = read_pcd()
        points = np.fromfile(f_path + str(i).zfill(6) + ".bin", dtype=np.float32, count=-1).reshape((-1, 4))

        # points = get_unground_points(points, img)
        point_cloud = o3d.geometry.PointCloud()

        """voxels, coordinates, num_points_per_voxel, voxel_point_mask, voxel_num"""
        t1 = time.time()
        res = voxel_generator.generate(points, max_voxels=90000)

        voxels = res["voxels"]
        num_points = res["num_points_per_voxel"]
        coords = res["coordinates"]
        coords = res["coordinates"]
        # add batch idx to coords
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
        example = {
            "anchors": anchors,
            "voxels": voxels,
            "num_points": num_points,
            "coordinates": coords
        }


        pred = net(example)[0]
        t2 = time.time()
        print("cost time : {}".format(round((t2 - t1) * 1000, 3)))
        # ���ӻ�
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
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        line_sets.append(line_range)
        custom_draw_geometry(point_cloud, line_sets)
        # name = pc_files[i].replace("bin", "npy")
        # np.save(save_path + name,
        #         np.concatenate((pred["box3d_lidar"].cpu().numpy(), pred["label_preds"].cpu().numpy().reshape((-1, 1))), 1))

    # print("average cost_time : {}".format(t_all / 1633 * 1000))

if __name__ == '__main__':

    config_path = '/data/second_dv/all.wjdata.semi.config'
    model_dir = 'dv'
    model_dir = "dv_kd"
    teacher_weight = "/data/second_dv/second/pytorch/dv/nice/voxelnet-13700.tckpt"
    train(config_path, model_dir, ckpt_path=teacher_weight, kd=True)    # kd train 
    train(config_path, model_dir, kd=False)    # normal train 

