# -*- coding: utf-8 -*-
# @Time    : 2/11/20 8:44 PM
# @Author  : Yuan Gong
# @Affiliation  : University of Notre Dame
# @Email   : yuangongfdu@gmail.com 
# @File    : test.py

import os
import constants
import data_loader
import model
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics
import scipy.special
import multiprocessing

def compute_eer(label, pred, positive_label=1, plot_path=None):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2

    if plot_path != None:
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(plot_path+'_roc.png')
        plt.close()
    return eer

def test(model=None, data_obj=None, device='cuda:0', record_path=None):

    # if provide path, then read the path
    if type(model) == str:
        net = torch.load(model)
    # if a model object is provided
    else:
        net = model
        net.eval()

    # if not specifically mentioned, test on all the data
    if data_obj == None:
        remasc_obj = data_loader.Remasc(os.path.join(constants.COMPLETE_PATH, 'meta.csv'), \
                                             os.path.join(constants.COMPLETE_PATH, 'data'), \
                                             BClass=True, \
                                             transform=torchvision.transforms.Compose(
                                                 [data_loader.AdjustChannel(1), data_loader.AdjustLength(16000), \
                                                  data_loader.NormScale()]))
    else:
        remasc_obj = data_obj

    remasc_loader = torch.utils.data.DataLoader(remasc_obj, batch_size=128, shuffle=False,  num_workers=int(multiprocessing.cpu_count()),
                                                pin_memory=True,
                                                drop_last=True)

    correct, total = 0, 0
    class_correct, class_total = list(0. for i in range(2)), list(0. for i in range(2))

    all_label = []
    all_pred = []

    with torch.no_grad():
        for data in remasc_loader:
            inputs, labels = data['waveform'].to(device), data['label'].to(device)
            outputs = net(inputs)
            preds_logit = outputs
            #print(preds_logit)
            # 1 is the dim, not compare with 1
            _, preds_discrete = torch.max(preds_logit, 1)
            total += labels.size(0)
            correct += (preds_discrete == labels).sum().item()

            all_label.extend(labels.tolist())
            # records the logits
            all_pred.extend(preds_logit.tolist())

            c = (preds_discrete == labels).squeeze()
            for i in range(128):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    acc = 100 * correct / total
    print('Overall Accuracy: {:.3f} ({:d}/{:d})'.format(100 * correct / total, correct, total))
    for i in range(2):
        print('Class [{:d}] Accuracy: {:.3f} ({:.0f}/{:.0f})'.format(i, 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))

    # convert to true label
    all_pred_discrete = np.argmax(all_pred, axis=1)
    # convert to softmax (probaility of the class 1)
    all_pred_softmax = scipy.special.softmax(all_pred, axis=1)
    all_pred_pos_prob = all_pred_softmax[:, 1]

    f1_score = sklearn.metrics.f1_score(all_label, all_pred_discrete, average='macro')
    eer = compute_eer(all_label, all_pred_pos_prob)

    if record_path != None:
        np.savetxt(record_path + '_label.csv', all_label[0:128], delimiter=',')
        np.savetxt(record_path + '_logits.csv', all_pred[0:128], delimiter=',')
        np.savetxt(record_path + '_posprob.csv', all_pred_pos_prob[0:128], delimiter=',')
        np.savetxt(record_path + '_softmax.csv', all_pred_softmax[0: 128], delimiter=',')
        _ = compute_eer(all_label, all_pred_pos_prob, plot_path=record_path)

    return acc, f1_score, eer

if __name__ == '__main__':
    acc, f1_score, eer = test(os.path.join(constants.EXP_PATH, '1.mdl'), data_obj=None)
