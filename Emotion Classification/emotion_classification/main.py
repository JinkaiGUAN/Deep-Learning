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
from torch.utils.data import DataLoader
from mydataset import EmotionDataset  # make the project being source root
from mobilenet_v2 import mobilenet_v2

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

    label_name = {'none': 0, 'pouting': 1, 'smile': 2, 'openmouth': 3}

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

    # check the crop size, it must satisfy the net input
    # Data preprocessing
    train_transform = transforms.Compose(
        [transforms.Resize(112), transforms.RandomCrop(96),
         transforms.ToTensor(),
         transforms.Normalize(mean=norm_mean, std=norm_std)])
    val_transform = transforms.Compose(
        [transforms.Resize(112), transforms.RandomCrop(96),
         transforms.ToTensor,
         transforms.Normalize(mean=norm_mean, std=norm_std)])
    # Dataset
    train_dataset = EmotionDataset(data_root=os.path.join(ROOT, 'Data'),
                                   mode='train', split_ratio=0.9, rng_seed=620,
                                   transform=train_transform)
    val_dataset = EmotionDataset(data_root=os.path.join(ROOT, 'Data'),
                                 mode='val', split_ratio=0.9, rng_seed=620,
                                 transform=val_transform)
    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32,
                                  shuffle=True, num_workers=3)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32,
                                shuffle=True, num_workers=3)

    # ============================= 2. Model ================================= #
    model = mobilenet_v2()

    # ============================ 3. Loss function ========================= #
    criterion = nn.CrossEntropyLoss()

    # ============================= 4. Optimizer ============================ #
    optimizer = optim.SGD(params=model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=1e-4)
    sceduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    # ============================= 5. Training ============================= #
