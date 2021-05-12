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
from src.mydataset import EmotionDataset  # make the project being source root
from src.mobilenet_v2 import mobilenet_v2
from src.utils import ModelTrainer, plot_comparison

import json
from datetime import datetime

if __name__ == '__main__':

    # config
    ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
    state_dict_wts_path = os.path.join(ROOT, "emotion_classification", "src",
                                       'mobilenet_v2-b0353104.pth')

    # Time display
    current_time = datetime.now()
    try:
        time_str = datetime.strftime(current_time, fmt='%Y-%m-%d-%H-%M')
    except:
        time_str = datetime.strftime(current_time, '%Y-%m-%d-%H-%M')

    # create the folder storing all the results.
    log_dir_path = os.path.join(ROOT, 'results', time_str)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    label_name = {'none': 0, 'pouting': 1, 'smile': 2, 'openmouth': 3}

    num_classes = len(label_name)
    MAX_EPOCH = 10
    LR = 0.01  # learning rate
    milestones = [5, 8]  # divide it by 10 at 5th and 8th epoch

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
         transforms.ToTensor(),
         transforms.Normalize(mean=norm_mean, std=norm_std)])
    # Dataset
    train_dataset = EmotionDataset(data_root=os.path.join(ROOT, 'Data'),
                                   mode='train', split_ratio=0.9, rng_seed=620,
                                   transform=train_transform)
    val_dataset = EmotionDataset(data_root=os.path.join(ROOT, 'Data'),
                                 mode='val', split_ratio=0.9, rng_seed=620,
                                 transform=val_transform)
    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=64,
                                shuffle=True, num_workers=4)

    # Gain the GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ============================= 2. Model ================================= #
    # model = mobilenet_v2(pretrained=True)
    model = mobilenet_v2()
    pretrained_state_dict_wts = torch.load(state_dict_wts_path)
    model.load_state_dict(pretrained_state_dict_wts)

    print(model)

    input_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_ftrs, out_features=num_classes,
                                    bias=True)

    model.to(device=device)

    # ============================ 3. Loss function ========================= #
    criterion = nn.CrossEntropyLoss()

    # ============================= 4. Optimizer ============================ #
    optimizer = optim.SGD(params=model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

    # ============================= 5. Training ============================= #
    loss_log = {'train': [], 'val': []}
    acc_log = {'train': [], 'val': []}
    best_acc, best_epoch = 0.0, 0

    for i in range(MAX_EPOCH):
        # train the model
        loss_train, acc_train = ModelTrainer.train(dataloader=train_dataloader,
                                                   model=model,
                                                   criterion=criterion,
                                                   optimizer=optimizer,
                                                   device=device, epoch_id=i,
                                                   max_epoch=MAX_EPOCH)
        # evaluate the model
        loss_val, acc_val = ModelTrainer.val(val_dataloader, model, criterion,
                                             device, epoch_id=i,
                                             max_epoch=MAX_EPOCH)
        scheduler.step()

        # print the information
        print(
            "Epoch[{:0>3}/{:0>3}] \t Train Loss: {:.4f} \t Val Loss: {:.4f} \t "
            "Train Acc: {:.4f} \t Val Acc: {:.4f}".format(
                i + 1, MAX_EPOCH, loss_train, loss_train, acc_train, acc_train))
        print('-' * 10)

        loss_log['train'].append(loss_train)
        loss_log['val'].append(loss_val)
        acc_log['train'].append(acc_train)
        acc_log['val'].append(acc_val)

        # save model weights
        if best_acc < acc_val and (i > (MAX_EPOCH) / 2):
            best_acc, best_epoch = acc_val, i + 1

            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': i + 1, 'best_acc': best_acc}
            path_checkpoint = os.path.join(log_dir_path, 'chechpoint_best.pkl')
            torch.save(checkpoint, path_checkpoint)

    log = {'loss': loss_log, 'acc': acc_log}
    json_str = json.dumps(log, ensure_ascii=False, indent=4)
    json_log_path = os.path.join(log_dir_path, 'log.json')
    with open(json_log_path, 'w') as json_file:
        json_file.write(json_str)

    # plot the graph
    plot_comparison(json_log_path)

    end_time_str = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')
    print('End time ', end_time_str)
