# -*- coding: utf-8 -*-
# @Time    : 2/5/20 3:17 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : train.py

import os
import constants
import data_loader
import model
import math
import numpy as np
import time
import test
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
from warmup_scheduler import GradualWarmupScheduler

"""
train_data: should be a pytorch data object
base_model: the model continue to train
"""
def train(bsize, lr, epoch_num=100, train_data=None, test_data=None, device='cuda:0', base_model=None, record_path=None):

    # if no model is provide, train from scratch
    if base_model == None:
        net = model.model_init()
        net = net.to(device)
    else:
        # if a model path is provided
        if type(base_model) == str:
            net = torch.load(base_model).to(device)
        # if a model obj is provided
        else:
            net = base_model

    # prepare the data, pass the pytorch data object, if not mentioned, test on the complete set0
    if train_data == None:
        train_data = data_loader.Remasc(os.path.join(constants.COMPLETE_PATH, 'meta.csv'), \
                                         os.path.join(constants.COMPLETE_PATH, 'data'), \
                                         BClass=True, \
                                         transform=torchvision.transforms.Compose(
                                             [data_loader.AdjustChannel(1), data_loader.AdjustLength(16000), \
                                              data_loader.NormScale()]))

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True, num_workers=int(multiprocessing.cpu_count()),
                                                    pin_memory=True,
                                                drop_last=True)

    # use to find the number of class, to balance the class
    train_label = [x['label'] for x in train_data]

    # create your optimizer, define the loss
    #optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=20, after_scheduler=scheduler)

    criterion = nn.CrossEntropyLoss()
    # note by intension the weight is set as REVERSE sample count
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([float(train_label.count(1)), float(train_label.count(0))]).to(device))
    err_tracker = []

    start_time = time.time()
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print('--------------------------------')
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['waveform'].to(device), data['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()


        running_loss /= len(train_data)

        if epoch % 10 == 0:
            save_path_tr = os.path.join(record_path, str(epoch) + '_tr_')
            acc_train, f1_train, eer_train = test.test(net, data_obj=train_data, device=device, record_path=save_path_tr)


        if epoch % 1 == 0:
            save_path_te = os.path.join(record_path, str(epoch) + '_te_')
            acc, f1, eer = test.test(net, data_obj=test_data, device=device, record_path=save_path_te)

        err_tracker.append([running_loss, acc, acc_train, f1, f1_train, eer, eer_train])

        end_time = time.time()
        print('Epoch {:d} loss: {:.3f}, acc: {:.3f}, acc_train: {:.3f}, f1: {:.3f}, f1_train: {:.3f}, eer: {:.3f}, eer_train: {:.3f}, time: {:.3f}'\
              .format(epoch, running_loss, acc, acc_train, f1, f1_train, eer, eer_train, end_time-start_time))
        start_time = end_time

        scheduler_warmup.step()
        for param_group in optimizer.param_groups:
            print('learning rate : ' + str(param_group['lr']))

if __name__ == '__main__':
    net, loss = train(128, 1e-3)
    torch.save(net, constants.EXP_PATH + '1.mdl')