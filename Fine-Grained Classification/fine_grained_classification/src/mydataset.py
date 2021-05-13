# -*- coding: UTF-8 -*-
"""
@Project : Project 
@File    : mydataset.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 13/05/2021 21:47 
@Brief   : 
"""
import os
import cv2
import torch
from torch.utils.data import Dataset

import random


class BirdDataset(Dataset):
    """

    :arg
        @root (string): the absolute root path of the data folder.
    """

    def __init__(self, root, mode='train', random_seed=0, split_ratio=0.9,
                 transform=None):
        self.data_root = root
        self.mode = mode
        self.rng = random_seed
        self.split_ratio = split_ratio
        self.transform = transform
        self.data_info = self._get_data_info()
        self.num_classes = self._get_class_num()

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        img_path, label = self.data_info[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label

    def _get_data_info(self):
        path_list = []
        label_list = []
        for folder_name in os.listdir(self.data_root):
            label = int(folder_name.split('.')[0]) - 1
            for img_name in os.listdir(
                    os.path.join(self.data_root, folder_name)):
                if not img_name.startswith('.'):
                    path_list.append(
                        os.path.join(self.data_root, folder_name, img_name))
                    label_list.append(label)
        data_info_all = [(img_path, label) for img_path, label in
                         zip(path_list, label_list)]

        # split the data
        random.seed(self.rng)
        random.shuffle(data_info_all)
        split_idx = int(self.split_ratio * len(data_info_all))

        if self.mode == 'train':
            data_info = data_info_all[:split_idx]
        elif self.mode == 'val':
            data_info = data_info_all[split_idx:]
        else:
            raise Exception("Please input valid mode name (train/val) when" \
                            " creating dataset")

        return data_info

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception(
                "\nPlease input the valid data path in creating dataset." \
                " Path: {}".format(self.data_root))
        return len(self.data_info)

    def _get_class_num(self):
        if len(os.listdir(os.path.join(self.data_root))) == 0:
            raise Exception(
                "\nPlease input the valid data path in creating dataset." \
                " Path: {}".format(self.data_root))
        return len(os.listdir(os.path.join(self.data_root)))



if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    dataset = BirdDataset(os.path.join(base_path, 'data'), mode='train',
                          random_seed=620, split_ratio=0.9, transform=None)
