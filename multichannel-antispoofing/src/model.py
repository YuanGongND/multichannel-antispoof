# -*- coding: utf-8 -*-
# @Time    : 2/6/20 11:49 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com
# @File    : model.py

import constants
import data_loader
import os
import math
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# initialize the model
def model_init(channel_num = 1, p_num=20, sr = 44100, audio_len=0.2, frame_time=0.010):
    # audio info
    frame_size = int(sr*frame_time)
    print(frame_size)

    # time-convolution (tconv) layer parameters
    tconv_ksize = int(frame_size/3.5*2.5)
    print(tconv_ksize)

    # frequency-onvolution (fconv) layer parameters
    fconv_nfilter, fconv_ksize = 256, 8

    # fully connected layer parameter
    pool_size, fc1_dim = 3, 256

    # LSTM layer parameters
    lstm_dim, lstm_layer = 832, 3
    #lstm_dim, lstm_layer = 256, 3

    # try to dynamically get the input size of fc1
    test_net = AudioNet(channel_num, p_num, fconv_nfilter, tconv_ksize, fconv_ksize, frame_size, pool_size, fc1_dim,
                         lstm_dim, lstm_layer, 10)
    test_input = torch.rand([1, channel_num, int(sr * audio_len)])
    fc1_indim = test_net(test_input, test=True)
    del test_net
    #print(fc1_indim)
    net = AudioNet(channel_num, p_num, fconv_nfilter, tconv_ksize, fconv_ksize, frame_size, pool_size, fc1_dim,
                         lstm_dim, lstm_layer, fc1_indim[1])
    return net

class AudioNet(nn.Module):
    def __init__(self, channel_num, p_num, fconv_nfilter, tconv_ksize, fconv_ksize, frame_size, pool_size, fc1_dim, lstm_dim, lstm_layer, fc1_indim):
        super(AudioNet, self).__init__()

        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.channel_num = channel_num
        self.frame_size = frame_size

        self.front_tconv = [None] * channel_num
        for i in range(channel_num):
            #self.front_tconv[i] = nn.Conv1d(1, p_num, tconv_ksize, padding = int(tconv_ksize / 2))
            self.front_tconv[i] = nn.Conv1d(1, p_num, tconv_ksize)
        self.front_tconv = nn.ModuleList(self.front_tconv)

        self.fconv = nn.Conv1d(1, fconv_nfilter, fconv_ksize, padding = int(fconv_ksize / 2))

        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)

        # fully connected layer before LSTMs
        #
        self.fc1 = nn.Linear(fc1_indim, fc1_dim)

        self.lstm = nn.LSTM(fc1_dim, hidden_size=lstm_dim, num_layers=lstm_layer, batch_first=True)

        #self.fc2 = nn.Linear(lstm_dim, 512)
        self.fc3 = nn.Linear(lstm_dim, 2)


    def forward(self, x, test=False):
        # the real input size: torch.Size([64, 2, 16000])
        batch_size = x.size()[0]
        channel_num = x.size()[1]
        audio_len = x.size()[2]

        # verify the network matches with the
        assert channel_num == self.channel_num

        # drop the last point to fit the fiter size
        x = x[:, :, 0: math.floor(audio_len/self.frame_size)*self.frame_size]
        # time convolution
        x_channel = [None] * channel_num
        for i in range(channel_num):
            x_channel[i] = x[:, i, :]
            # hop size is 10ms, which is 1/3.5 of the frame size
            x_channel[i] = x_channel[i].unfold(1, self.frame_size, int(self.frame_size))
            subseq_num = x_channel[i].shape[1]
            x_channel[i] = x_channel[i].reshape([batch_size * subseq_num, 1, self.frame_size])
            x_channel[i] = self.front_tconv[i](x_channel[i])

        # sum up channel, max pooling, and nonlinear transform
        x = torch.sum(torch.stack(x_channel), dim=0)
        x = torch.max(x, dim=2)[0]
        x = F.relu(x)
        #x = torch.log(x + 0.01)

        # frequency convolution, start of the CLDNN net
        x = x.reshape([x.size(0), 1, x.size(1)])
        x = self.fconv(x)
        x = F.relu(x)

        # should only pool on the frequency coordination
        x = self.maxpool(x)
        x = x.reshape([x.size(0), int(x.size(1)*x.size(2))])

        if test == True:
            return x.size()

        # first linear layer
        x = self.fc1(x)
        x = F.relu(x)

        # LSTM layers
        x = x.reshape([batch_size, subseq_num, x.size(1)])
        x = self.lstm(x)[0][:, -1, :]

        # second linear layer
        #x = self.fc2(x)

        # DO NOT ADD ReLu to the last layer, it will block the negative output
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    net = model_init(channel_num=6, p_num=20)
    print(net)
    a = torch.rand([2, 6,14700])
    o = net(a)
