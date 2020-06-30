# -*- coding: utf-8 -*-
# @Time    : 2/29/20 8:06 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : uniform_sample_rate.py

import os
import shutil
import scipy.io.wavfile
import constants

set_list = ['core', 'complete', 'eval']
sample_rate_list = [4000, 8000, 16000, 22050, 44100]

for set in set_list:
    for sr in sample_rate_list:
        print('working on {}: {}'.format(set, sr))
        original_path = constants.PROJ_PATH + '/data/{}/data'.format(set)
        target_path =   constants.PROJ_PATH + '/data/{}_{}/data'.format(set, sr)

        # if the set is not processed
        if not os.path.exists(target_path):
            os.makedirs(target_path)

            audio_list = os.listdir(original_path)
            resample_count = 0

            for i in range(len(audio_list)):
                file_name = audio_list[i]
                sample_rate, _ = scipy.io.wavfile.read(os.path.join(original_path, file_name))
                # if sample rate is 44100, simply copy it to the new directory
                if sample_rate == sr:
                    shutil.copyfile(os.path.join(original_path, file_name), os.path.join(target_path, file_name))
                # if the sample rate is 16000 (only for device 4 AMlOGIC, resample it to 44100)
                else:
                    os.system('sox {} -r {} -G {}'.format(os.path.join(original_path, file_name), str(sr), os.path.join(target_path, file_name)))
                    resample_count += 1
                    if resample_count % 100 == 0:
                        print(resample_count)