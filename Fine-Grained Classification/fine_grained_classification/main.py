# -*- coding: UTF-8 -*-
"""
@Project : Project 
@File    : main.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 13/05/2021 21:25 
@Brief   : 
"""
import os
from datetime import datetime
import torch
from torch import optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader

from src.mydataset import BirdDataset
import json


if __name__ == '__main__':
    # ======================= Config ==================================== #
    # Gain the ROOT of this project, related to the main.py
    ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))

    current_time = datetime.now()
    current_time_str = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')

    log_path = os.path.join(ROOT, 'results', current_time_str)
    if os.path.exists(log_path):
        os.mkdir(log_path)

    # List some important configuration parameters.
    MAX_EPOCH = 10
    LR = 0.01
    milestones = [4, 8]

    # ============================= 1. Data ================================ #
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # make transforms
    train_transforms = transforms.Compose(
        [transforms.Resize(480), transforms.RandomCrop(448),
         transforms.ToTensor(),
         transforms.Normalize(mean=norm_mean, std=norm_std)])
    val_transforms = transforms.Compose(
        [transforms.Resize(480), transforms.RandomCrop(448),
         transforms.ToTensor(),
         transforms.Normalize(mean=norm_mean, std=norm_std)])

    # gain the datasets
    train_dataset = BirdDataset(os.path.join(ROOT, 'data'), mode='train',
                                random_seed=620, split_ratio=0.9,
                                transform=train_transforms)
    val_dataset = BirdDataset(os.path.join(ROOT, 'data'), mode='val',
                              random_seed=620, split_ratio=0.9,
                              transform=val_transforms)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)

    # class-number
    num_classes = train_dataset.num_classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ================================ 2. Model ============================= #

