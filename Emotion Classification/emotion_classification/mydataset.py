# -*- coding: UTF-8 -*-
"""
@Project : Project 
@File    : mydataset.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 11/05/2021 15:48 
@Brief   : 
"""
import os
import torch
import cv2
from torch.utils.data import Dataset
import random


class EmotionDataset(Dataset):
    r"""The Dataset class for this the data.

    :argument
        @data_root (string): The data root absolute path.
        @mode (string): 'train' or 'val'.
        @split_ratio (float): the ratio that training set takes part of the
         whole dataset.
        @rng_seed (int): random seed.
        @transform (torchvision.transforms): transforms module in torchvision.

    :returns
        @img (np.array)
        @label (int)
    """

    def __init__(self, data_root, mode='train', split_ratio=0.9, rng_seed=0,
                 transform=None):
        super(EmotionDataset, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.split_ratio = split_ratio
        self.rng_seed = rng_seed
        self.transform = transform
        self.data_info = self._get_data_info()

    def __getitem__(self, idx):
        r"""Return the image and label of this index."""
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        img_path, label = self.data_info[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def _get_data_info(self):
        r"""Helper function to split the data into training and testing sets."""
        image_path_list = []
        image_label_list = []
        for folder in os.listdir(self.data_root):
            if folder[0].isdigit():
                # the folder is the data
                label = int(folder[0])
                for img in os.listdir(os.path.join(self.data_root, folder)):
                    if img.endswith('.jpg'):
                        img_path = os.path.join(self.data_root, folder, img)
                        image_path_list.append(img_path)
                        image_label_list.append(label)

        data_info = [(img, label) for img, label in
                     zip(image_path_list, image_label_list)]

        # Shuffle the data and split it into two parts, i.e., training and testing
        random.seed(self.rng_seed)
        random.shuffle(data_info)

        split_idx = int(len(data_info) * self.split_ratio)
        if self.mode == 'train':
            img_set = data_info[:split_idx]
        elif self.mode == 'val':
            img_set = data_info[split_idx:]
        else:
            raise Exception(
                "Please input valid mode (train / val) in creating Dataset.")

        return img_set


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_root = os.path.join(root, 'Data')
    dataset = EmotionDataset(data_root=data_root, mode='train', split_ratio=0.9,
                             rng_seed=620, transform=None)
    print(dataset[3])
