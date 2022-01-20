import glob
import math
import os
import random
import shutil
import subprocess
from pathlib import Path
from sys import platform

import numpy as np
import torch
import torch.nn as nn
import torchvision
import pdb
import torchplus

import torch.nn.functional as F


def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]],
                       dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                       dim=-1)
    return boxes1, boxes2


def compute_lost_KD(example, student_dict, teacher_dict):
    pdb.set_trace()
    reg_m = 0.0
    T = 10
    num_class = 4
    sin_error_factor = 1.0
    Lambda_cls, Lambda_box = 0.0001, 0.001
    criterion_st = torch.nn.KLDivLoss(reduction='sum')

    ft = torch.cuda.FloatTensor
    lcls, lbox = ft([0]), ft([0])
    
    stu_box_preds = student_dict["box_preds"]
    batch_size = int(stu_box_preds.shape[0])
    stu_box_preds = stu_box_preds.view(batch_size, -1, 7)
    stu_cls_preds = student_dict["cls_preds"]
    stu_cls_preds = stu_cls_preds.view(batch_size, -1, num_class)

    labels = example['labels']
    cared = labels >= 0

    cls_targets = labels * cared.type_as(labels)
    cls_targets = cls_targets.unsqueeze(-1)
    reg_targets = example['reg_targets']
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=stu_box_preds.dtype)

    tea_box_preds = teacher_dict["box_preds"]
    tea_box_preds = tea_box_preds.view(batch_size, -1, 7)
    tea_cls_preds = teacher_dict["cls_preds"]
    tea_cls_preds = tea_cls_preds.view(batch_size, -1, num_class)

    #pdb.set_trace()

    stu_box_preds, stu_reg_targets = add_sin_difference(stu_box_preds, reg_targets, stu_box_preds[..., 6:7],
                                                        reg_targets[..., 6:7],
                                                        sin_error_factor)

    tea_box_preds, tea_reg_targets = add_sin_difference(tea_box_preds, reg_targets, tea_box_preds[..., 6:7],
                                                        reg_targets[..., 6:7],
                                                        sin_error_factor)

    ###loss_loc####

    l2_dis_s = (stu_box_preds - stu_reg_targets).pow(2).sum(1)
    l2_dis_s_m = l2_dis_s + reg_m
    l2_dis_t = (tea_box_preds - tea_reg_targets).pow(2).sum(1)
    l2_num = l2_dis_s_m > l2_dis_t
    lbox += l2_dis_s[l2_num].sum()


    ###loss_cls####
    lcls += criterion_st(nn.functional.log_softmax(stu_cls_preds / T, dim=1),
                         nn.functional.softmax(tea_cls_preds / T, dim=1)) * (T * T) / batch_size

    soft_loss = lcls * Lambda_cls + lbox * Lambda_box

    return soft_loss


def compute_lost_KD2(example, student_dict, teacher_dict):
    T = 10
    num_class = 7
    sin_error_factor = 1.0
    Lambda_cls, Lambda_box = 0.0001, 0.001
    criterion_st = torch.nn.KLDivLoss(reduction='sum')
    ft = torch.cuda.FloatTensor
    lcls, lbox = ft([0]), ft([0])
    stu_box_preds = student_dict["box_preds"]
    batch_size = int(stu_box_preds.shape[0])
    stu_box_preds = stu_box_preds.view(batch_size, -1, 7)
    stu_cls_preds = student_dict["cls_preds"]
    stu_cls_preds = stu_cls_preds.view(batch_size, -1, num_class)

    reg_targets = example['reg_targets']


    tea_box_preds = teacher_dict["box_preds"]
    tea_box_preds = tea_box_preds.view(batch_size, -1, 7)
    tea_cls_preds = teacher_dict["cls_preds"]
    tea_cls_preds = tea_cls_preds.view(batch_size, -1, num_class)


    stu_box_preds, stu_reg_targets = add_sin_difference(stu_box_preds, reg_targets, stu_box_preds[..., 6:7],
                                                        reg_targets[..., 6:7],
                                                        sin_error_factor)

    tea_box_preds, tea_reg_targets = add_sin_difference(tea_box_preds, reg_targets, tea_box_preds[..., 6:7],
                                                        reg_targets[..., 6:7],
                                                        sin_error_factor)

    ###loss_loc####
    l2_dis = (stu_box_preds - tea_box_preds).pow(2).sum(1)
    lbox += l2_dis.sum()
    ###loss_oc####
    lcls += criterion_st(nn.functional.log_softmax(stu_cls_preds / T, dim=1),
                         nn.functional.softmax(tea_cls_preds / T, dim=1)) * (T * T) / batch_size

    soft_loss = lcls * Lambda_cls + lbox * Lambda_box

    return soft_loss[0]
