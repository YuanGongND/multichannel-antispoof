# -*- coding: utf-8 -*-
# @Time    : 2/5/20 2:50 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : helper.py

import constants
import os
import torch
import model as mdl
import data_loader
import shutil
import numpy as np

def save_model(model, path, model_name=None):
    if model_name == None:
        path = os.path.join(path, 'model.mdl')
    else:
        path = os.path.join(path, model_name + '.mdl')
    torch.save(model, path)
    print('model saved in {}'.format(path))

def load_model(model_path):
    return torch.load(model_path)

def expname_to_path(new_dir):
    new_dir = os.path.join(constants.EXP_PATH, new_dir)
    while os.path.exists(new_dir) == True:
        new_dir += '_1'
    os.mkdir(new_dir)
    return new_dir

def copyfile(new_dir):
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)), new_dir + '/src')
    print('Source files saved in {}'.format(new_dir + '/src'))

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == '__main__':
    pass