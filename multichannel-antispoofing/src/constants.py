# -*- coding: utf-8 -*-
# @Time    : 2/10/20 5:21 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : constants.py

# change this path to your own path
PROJ_PATH = '/home/ndmobilecomp/remasc/multichannel-antispoofing'

CORE_PATH = PROJ_PATH + '/data/core/'
COMPLETE_PATH = PROJ_PATH + '/data/complete/'
# evaluation = complete - core
EVAL_PATH = PROJ_PATH + '/data/eval/'
EXP_PATH = PROJ_PATH + '/exp/'
DUMMY_PATH = PROJ_PATH + '/data/dummy/'
TMP_PATH = PROJ_PATH + '/exp/tmp_result/'
MDL_PATH = PROJ_PATH + '/model/'
SUMMARY_PATH = PROJ_PATH + '/exp/exp_summary.csv'
OLD_SUMMARY_PATH = PROJ_PATH + '/exp/old_summary.csv'

# map from protocal device id to device name
MIC = {1: 'AIY', 2: 'RES_4', 3: 'RES_CORE', 4: 'AMLOGIC'}

# the channel id used when n channel is tested for each device
MIC_ARRAY_CHANNEL = {'AIY': {1: [0], 2: [0, 1]},
                    'RES_4': {1: [0], 2: [0, 3], 3: [0, 1, 3], 4: [0, 1, 2, 3]},
                    #'RES_4': {1: [0], 2: [0, 1], 3: [0, 1, 2], 4: [0, 1, 2, 3]},
                    'RES_CORE': {1: [0], 2: [0,3], 3: [0,1,3], 4: [0,1,3,4], 5: [0,1,2,3,4], 6: [0,1,2,3,4,5]},
                    'AMLOGIC': {1: [0], 2: [0,3], 3: [0,1,3], 4: [0,1,3,4], 5: [0,1,2,3,4], 6: [0,1,2,3,4,5], 7: [0,1,2,3,4,5,6]}} # the center mic is added at last