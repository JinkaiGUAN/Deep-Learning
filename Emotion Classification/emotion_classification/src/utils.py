# -*- coding: UTF-8 -*-
"""
@Project : Project 
@File    : utils.py
@IDE     : PyCharm 
@Author  : Peter
@Date    : 12/05/2021 13:46 
@Brief   : Training process
"""
import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "axes.titlesize": 24
}
rcParams.update(config)


class ModelTrainer(object):
    r"""

    :arg
        @dataloader (torchvision.DataLoader):
    """

    @staticmethod
    def train(dataloader, model, criterion, optimizer, device, epoch_id,
              max_epoch):
        model.train()  # set the model to training mode

        loss_list = []  # loss per batch
        acc_list = []  # accuracy per batch

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # clear the gradients
            optimizer.zero_grad()

            outputs = model(inputs)  # learn what the ouputs are
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # ----------------- statistics -------------------------------------- #
            _, preds = torch.max(outputs, dim=1)
            # actually, the loss is an average one, a scalar.
            loss_list.append(float(loss.item()))
            acc_list.append(
                float(np.sum((preds == labels).cpu().numpy())) / len(
                    labels.cpu().numpy()))

            # print information every 10 batches
            if i % 50 == 0 and i > 0:
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] \t Batch[{:0>3}/{:0>3}] \t Loss: {:.4f} \t Acc: {:.4f}".format(
                        epoch_id + 1, max_epoch, i + 1,
                        len(dataloader) + 1, np.mean(loss_list),
                        np.mean(acc_list)))

        # return the final loss and accuracy for this epoch.
        return np.mean(loss_list), np.mean(acc_list)

    @staticmethod
    def val(dataloader, model, criterion, device, epoch_id, max_epoch):
        model.eval()
        print("-" * 10, 'Val', '-' * 10)

        loss_list, acc_list = [], []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            loss_list.append(loss.item())
            acc_list.append(
                float(np.sum((preds == labels).cpu().numpy())) / len(
                    labels.cpu().numpy()))
            # print("Iteration {}/{}".format(i, len(dataloader)))

        return np.mean(loss_list), np.mean(acc_list)


def plot_comparison(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data_dic = json.load(f)
    loss, acc = data_dic['loss'], data_dic['acc']

    fig = plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.plot([i + 1 for i in range(len(loss['train']))], loss['train'],
             label='Training')
    plt.plot([i + 1 for i in range(len(loss['val']))], loss['val'], label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([i + 1 for i in range(len(acc['train']))], acc['train'],
             label='Training')
    plt.plot([i + 1 for i in range(len(acc['val']))], acc['val'], label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout(h_pad=2)

    log_root = os.path.dirname(json_file_path)
    plt.savefig(os.path.join(log_root, 'plot.svg'), dpi=1600)


if __name__ == '__main__':
    ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    json_path = os.path.join(ROOT, 'results', '2021-05-12-17-10', 'log.json')
    plot_comparison(json_path)
