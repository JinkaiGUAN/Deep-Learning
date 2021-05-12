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
import numpy as np


class ModelTrainer(object):
    r"""

    :arg
        @dataloader (torchvision.DataLoader):
    """

    @staticmethod
    def train(dataloader, model, criterion,  optimizer, device, epoch_id, max_epoch):
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
            acc_list.append(float(np.sum((preds == labels).cpu().numpy())) / len(labels.cpu().numpy()))

            # print information every 10 batches
            if i % 10:
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] \t Batch[{:0>3}/{:0>3}] \t Loss: {:.4f} \t Acc: {:.4f}".format(
                        epoch_id + 1, max_epoch, i + 1,
                        len(dataloader) + 1, np.mean(loss_list), np.mean(acc_list)))

        # return the final loss and accuracy for this epoch.
        return np.mean(loss_list), np.mean(acc_list)


def plot_comparison(data):
    pass




