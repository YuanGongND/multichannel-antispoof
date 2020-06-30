# -*- coding: utf-8 -*-
# @Time    : 1/27/20 10:05 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com
# @File    : data_loader.py

import os, sys
import constants
import torch
import numpy as np
import torchaudio, torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
import librosa
import time

# define the channel number kept
class AdjustChannel(object):
    def __init__(self, channel_num):
        self.channel_num = channel_num

    def __call__(self, sample):
        waveform = sample['waveform']
        device = sample['device']
        if waveform.size()[0] >= self.channel_num:
            use_channel = constants.MIC_ARRAY_CHANNEL[constants.MIC[device]][self.channel_num]
            waveform = waveform[use_channel, :]
            sample['waveform'] = waveform
        else:
            #print(int(self.channel_num-waveform.size()[0]))
            raise Exception('cannot get channel more than original channel numbers')
        return sample

# this is a fake multichannel input that the content of all channel are that of the first channel, use for ablation study.
class AdjustChannel_Dummy(object):
    def __init__(self, channel_num):
        self.channel_num = channel_num

    def __call__(self, sample):
        waveform = sample['waveform']
        waveform = waveform[0, :].repeat(self.channel_num, 1)
        #print(waveform.size)
        sample['waveform'] = waveform
        return sample

# cut or pad to a specific audio length (in number of data points)
class AdjustLength(object):
    def __init__(self, audio_length):
        self.audio_length = audio_length

    def __call__(self, sample):
        waveform = sample['waveform']
        if waveform.size()[1] > self.audio_length:
            waveform = waveform[:, 0: self.audio_length]
        else:
            pad_len = self.audio_length - waveform.size()[1]
            #print(pad_len)
            pad_op = torch.nn.ZeroPad2d([0, pad_len, 0, 0])
            waveform = pad_op(waveform)
        sample['waveform'] = waveform
        return sample

# normalize the signal scale that has the max amplitude of 1 (-1), i.e., signal = 1 / max(signal) * signal
class NormScale(object):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, sample):
        waveform = sample['waveform']
        max_scale = max(abs(waveform.max()), abs(waveform.min()))
        waveform = (1 / max_scale) * waveform * self.scale
        sample['waveform'] = waveform
        return sample

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# main dataset class
class Remasc(Dataset):
    def __init__(self, meta_path, data_path, BClass=True, env='all', device='all', transform=AdjustLength(16000)):
        self.meta = np.loadtxt(meta_path, dtype='str', delimiter=',')
        #print(len(self.meta))
        self.meta = [([x[0].strip()] + list(x[1:])) for x in self.meta]
        self.file_list = [x.strip() for x in os.listdir(data_path)]
        self.data_path = data_path
        self.transform = transform
        self.BClass = BClass

        # only keep replayed (3) and genuine recording (2)
        self.bmeta = []
        for i in range(len(self.meta)):
            if int(self.meta[i][1]) > 1:
                self.bmeta.append(self.meta[i])
        if BClass == True:
            self.meta = self.bmeta

        # only keep selected device, input should be a list, element value range from [1,2,3,4]
        if device != 'all':
            dmeta = []
            for i in range(len(self.meta)):
                # index 7 is the device
                if int(self.meta[i][7]) in device:
                    dmeta.append(self.meta[i])
            self.meta = dmeta

        # only keep selected environment, input should be a list, element value range from [1,2,3,4]
        if env != 'all':
            emeta = []
            for i in range(len(self.meta)):
                # index 3 is the environment id
                if int(self.meta[i][3]) in env:
                    emeta.append(self.meta[i])
            self.meta = emeta

        # check if all files in the meta information exist in the data folder
        self.missing_file = []
        for i in range(0, len(self.meta)):
            cur_filename = self.meta[i][0] + '.wav'
            if cur_filename.strip() not in self.file_list:
                self.missing_file.append(cur_filename.strip())
                print(',' + cur_filename)
        print('Successfully load {} files, miss {} files.'.format(len(self.meta), len(self.missing_file)))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item_meta = self.meta[idx]
        audio_name = item_meta[0] + '.wav'
        #audio_name = audio_name.strip()
        audio_path = os.path.join(self.data_path, audio_name)
        with HiddenPrints():
            print('test')
            waveform, sample_rate = torchaudio.load(audio_path, normalization=False)
        if self.BClass == False:
            # adjust to source(0), genuine (1), and replayed (2)
            audio_label = int(item_meta[1]) - 1
        else:
            # only genuine (2) and replayed (3), adjust to 0 and 1
            audio_label = int(item_meta[1]) - 2

        device = int(item_meta[7])
        environment = int(item_meta[3])

        sample = {'waveform': waveform, 'sample_rate': sample_rate, 'label': audio_label, 'device': device, 'environment': environment}
        if self.transform:
            sample = self.transform(sample)
        return sample

# conduct filter-and-sum beamforming
class Beamforming(object):
    def __init__(self, output_size):
        pass

    def __call__(self, sample):
        pass

# Only select label with genuine and replayed
class SelectSample():
    def select_sample():
        pass

def print_summary():
    for i in [1, 2, 3, 4]:
        remasc_complete = Remasc(os.path.join(constants.CORE_PATH, 'meta.csv'),
                                 os.path.join(constants.CORE_PATH, 'data'), BClass=True, env=[1, 2, 3, 4], device=[i],
                                 transform=torchvision.transforms.Compose(
                                     [AdjustChannel_Dummy(1), AdjustLength(16000), NormScale()]))
        print('device {}: {}'.format(i, len(remasc_complete)))
        replay_c = 0
        genuine_c = 0
        for i in range(len(remasc_complete)):
            cur_label = int(remasc_complete[i]['label'])
            if cur_label == 0:
                genuine_c += 1
            else:
                replay_c += 1
        print('replay: {}, genuine: {}'.format(replay_c, genuine_c))

    for i in [1, 2, 3, 4]:
        remasc_complete = Remasc(os.path.join(constants.EVAL_PATH, 'meta.csv'),
                                 os.path.join(constants.EVAL_PATH, 'data'), BClass=True, env=[1, 2, 3, 4], device=[i],
                                 transform=torchvision.transforms.Compose(
                                     [AdjustChannel_Dummy(1), AdjustLength(16000), NormScale()]))
        print('device {}: {}'.format(i, len(remasc_complete)))

        replay_c = 0
        genuine_c = 0
        for i in range(len(remasc_complete)):
            cur_label = int(remasc_complete[i]['label'])
            if cur_label == 0:
                genuine_c += 1
            else:
                replay_c += 1
        print('replay: {}, genuine: {}'.format(replay_c, genuine_c))

# sample usage
if __name__ == '__main__':

    # Example 1: load a single file 

    # load a set, note: order of the transform does matter
    remasc_complete = Remasc(os.path.join(constants.COMPLETE_PATH, 'meta.csv'), os.path.join(constants.COMPLETE_PATH, 'data'), BClass=True, env=[1,2,3,4], device=[3], transform=torchvision.transforms.Compose([AdjustChannel(5), AdjustLength(16000), NormScale()]))
    # randomly select an sample from the set
    sample = remasc_complete[256]
    # get the waveform
    sample_wav = sample['waveform']
    # get the label
    label = sample['label']
    plt.plot(sample_wav.T)
    plt.show()
    # print the max/min value of the seleted sample
    print(sample_wav.max(), sample_wav.min())

    # Example 2: load in batch for train/test

