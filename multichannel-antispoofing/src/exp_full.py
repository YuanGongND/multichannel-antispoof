# -*- coding: utf-8 -*-
# @Time    : 3/3/20 9:45 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : exp_full.py

import matplotlib as mpl
mpl.use('Agg')
import argparse
import os
import constants
import data_loader
import math
import model
import numpy as np
import time
import train
import test
import torch
import helper
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools
import random

# reference: https://zhuanlan.zhihu.com/p/76472385
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_data(device, channel_num, point_num, real_mulch, sr):
    if real_mulch == True:
        train_data = data_loader.Remasc(os.path.join(constants.CORE_PATH, 'meta.csv'), \
                                        os.path.join(constants.CORE_PATH + '_' + str(sr), 'data'), \
                                        BClass=True, \
                                        device=[device], \
                                        transform=torchvision.transforms.Compose(
                                            [data_loader.AdjustChannel(channel_num),
                                             data_loader.AdjustLength(point_num), \
                                             data_loader.NormScale()]))

        test_data = data_loader.Remasc(os.path.join(constants.EVAL_PATH, 'meta.csv'), \
                                       os.path.join(constants.EVAL_PATH + '_' + str(sr), 'data'), \
                                       BClass=True, \
                                       device=[device], \
                                       transform=torchvision.transforms.Compose(
                                           [data_loader.AdjustChannel(channel_num), \
                                            data_loader.AdjustLength(point_num), \
                                            data_loader.NormScale()]))
    else:
        train_data = data_loader.Remasc(os.path.join(constants.CORE_PATH, 'meta.csv'), \
                                        os.path.join(constants.CORE_PATH + '_' + str(sr), 'data'), \
                                        BClass=True, \
                                        device=[device], \
                                        transform=torchvision.transforms.Compose(
                                            [data_loader.AdjustChannel_Dummy(channel_num),
                                             data_loader.AdjustLength(point_num), \
                                             data_loader.NormScale()]))

        test_data = data_loader.Remasc(os.path.join(constants.EVAL_PATH, 'meta.csv'), \
                                       os.path.join(constants.EVAL_PATH + '_' + str(sr), 'data'), \
                                       BClass=True, \
                                       device=[device], \
                                       transform=torchvision.transforms.Compose(
                                           [data_loader.AdjustChannel_Dummy(channel_num), \
                                            data_loader.AdjustLength(point_num), \
                                            data_loader.NormScale()]))
    return train_data, test_data

def exp(exp_name, device='cuda:0'):
    exp_path = helper.expname_to_path(exp_name)
    helper.copyfile(exp_path)

    # the hyper-parameter list plan to evaluate, not, all combination will be tested, so the time needed grows fast with the number of parameters you want to test.
    bsize_list = [64]
    lr_list = [1e-5]
    # recording device, call it rdevice to avoid confusing with the gpu device
    rdevice_list = [1, 2, 3, 4]
    audio_len_list = [1.0]
    filter_num_list = [64]
    sr_list = [44100]
    # using real multichannel or fake multichannel model (for ablation study)
    mch_setting = [True]
    frame_time_list = [0.02]

    param_list = list(itertools.product(bsize_list, lr_list, rdevice_list, audio_len_list, filter_num_list, sr_list, mch_setting, frame_time_list))

    # TODO: channel number is a risk

    for param in param_list:
        bsize, lr, rdevice, audio_len, filter_num, sr, mch, frame_time = param

        channel_num_max = len(constants.MIC_ARRAY_CHANNEL[constants.MIC[rdevice]])
        point_num = int(audio_len * sr)

        # test model using all channels available
        #for channel_num in range(channel_num_max, channel_num_max + 1):
        
        # test model using different number of channels
        for channel_num in range(1, channel_num_max + 1):
            # if real multichannel
            train_data, test_data = init_data(rdevice, channel_num, point_num, mch, sr)
            test_name = '_'.join([str(x) for x in param]) + '_' + str(channel_num)
            os.mkdir(os.path.join(exp_path, test_name))

            f = open(os.path.join(exp_path, 'result.txt'), 'a')
            fc = open(os.path.join(exp_path, 'result.csv'), 'a')

            # initialize the model:
            net = model.model_init(channel_num=channel_num, p_num=filter_num, sr=sr, audio_len=audio_len, frame_time=frame_time)
            net = net.to(device)
            print(net)

            # train the model
            net, err = train.train(bsize, lr, epoch_num=100, train_data=train_data, test_data=test_data, device=device, base_model=net, record_path=os.path.join(exp_path, test_name))
            err_name = ['running_loss', 'acc', 'acc_train', 'f1', 'f1_train', 'eer', 'eer_train']

            err = np.array(err)
            for i in range(err.shape[1]):
                plt.plot(err[:, i], label=err_name[i])
            plt.legend()
            plt.ylim([0,1])
            plt.savefig(os.path.join(exp_path, test_name + ".png"))
            plt.close()

            err = np.array(err)
            np.savetxt(os.path.join(exp_path, test_name + '.csv'), err, delimiter=',')
            err = list(np.amin(err, 0))
            helper.save_model(net, exp_path, model_name=test_name)

            print(err)

            result = [str(x) for x in (list(param) + [channel_num] + list(err))]
            f.write(','.join(result) + '\n')
            fc.write(','.join(result) + '\n')
            print('batch_size: {}, lr: {}, channel_num: {}, loss: {}, acc: {}, acc_train: {}, f1: {}, f1_train: {}, eer: {}, eer_train: {}'.format(bsize, lr, channel_num, *err))
            f.close()
            fc.close()
            with open(constants.SUMMARY_PATH, 'a') as fd:
                fd.write(','.join(result) + '\n')
            del net
            torch.cuda.empty_cache()

if __name__ == '__main__':
    #mpl.use('Agg')

    plt.ioff()
    parser = argparse.ArgumentParser(description='Run Experiments.')
    parser.add_argument('-d', '--device', type=int, choices=[0,1], default=0, help='gpu to run the exp, should be 0 or 1')
    parser.add_argument('-n', '--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    setup_seed(args.seed)
    if args.device == 0:
        device = 'cuda:0'
    else:
        device = 'cuda:1'
    #torch.manual_seed(5)
    exp(args.name, device)