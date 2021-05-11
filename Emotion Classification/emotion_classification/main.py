# -*- coding: UTF-8 -*-
"""
@Project : Project 
@File    : main.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 11/05/2021 15:02 
@Brief   : 
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import numpy as np
from datetime import datetime


if __name__ == '__main__':

    # config
    ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
    current_time = datetime.now()
    time_str = datetime.strftime(current_time, fmt='%Y-%m-%d-%H-%I-%M')
    # create the folder storing all the results.
    log_dir_path = os.path.join(ROOT, 'results', time_str)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    label_name = {'none':0, 'pouting':1, 'smile':2, 'openmouth':3}

    num_classes = len(label_name)
    MAX_EPOCH = 10  # 182     # min: 2348 images 2348 / 32 = 73.5 epochs
    LR = 0.01  # learning rate
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [5, 8]  # divide it by 10 at 32k and 48k iterations

    # ======================== 1. Data ======================================= #
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([transforms.Resize(), transforms.RandomCrop(), transforms.ToTensor(), transforms.Normalize(mean=norm_mean, std=norm_std)])



