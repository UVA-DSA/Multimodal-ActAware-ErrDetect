import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
import pandas as pd
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, jaccard_score, precision_score, recall_score
from metrics import binary_evaluation, multiclass_evaluation
import pdb
from logger import CompleteLogger

def main():
    settings = ['LOSO']#'LOUO',
    tasks = ['All', 'Suturing','Knot_Tying','Needle_Passing']
    out1 = ['Bout','Cout','Dout','Eout','Fout','Gout','Hout','Iout']
    out2 = ['1out','2out','3out','4out','5out']
    win_len = 10
    stride = 6
    for setting in settings:
        test_f1_fold = []
        test_acc_fold = []
        test_precision_fold = []
        test_recall_fold = []
        test_auc_roc_fold = []
        test_jaccard_fold = []
        if setting == 'LOSO':
            outs = out2
        else:
            outs = out1
        

        for out in outs:
            data_root = "./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out)

            window_true_pred = pd.DataFrame(None,columns=['gest','true','pred','score'])

            current_gest_number = None
            
            with open(data_root+'results.csv', mode='r') as file:
                reader = csv.reader(file)
                gest_count = -1
                win_count = 0
                for row in reader:
                    gest_number = int(row[0])
                    err_label = int(row[1])
                    err_pred = int(row[2])
                    pred = [err_pred]

                    if gest_number != current_gest_number:
                        if gest_count>=0:
                            time_len = len(preds)
                            start = (time_len-win_len)%stride
                            for i in np.arange(start, time_len-win_len+stride, stride):
                                cur_preds = preds[i:i+win_len]
                                TorF = 1 if np.mean(cur_preds)>args.threshold else 0
                                window_true_pred.loc[win_count] = [current_gest_number, t, TorF, np.mean(cur_preds)]
                                win_count += 1
                        current_gest_number = gest_number
                        preds = [pred]
                        t = err_label
                        gest_count += 1
                    else:
                        preds.append(pred)
            #gest_true_pred.to_csv("./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out)+"gest_true_pred.csv")
            window_true_pred.to_csv("./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out)+"window_true_pred.csv")
            test_f1 = f1_score(window_true_pred['true'].astype(int), window_true_pred['pred'].astype(int), average='binary', pos_label=1)
            test_accuracy = accuracy_score( window_true_pred['true'].astype(int), window_true_pred['pred'].astype(int))
            test_precision = precision_score( window_true_pred['true'].astype(int), window_true_pred['pred'].astype(int), average='binary', pos_label=1)
            test_recall = recall_score(window_true_pred['true'].astype(int),window_true_pred['pred'].astype(int), average='binary', pos_label=1)
            test_jaccard = jaccard_score(window_true_pred['true'].astype(int),window_true_pred['pred'].astype(int), average='binary', pos_label=1)
            test_auc_roc = roc_auc_score(window_true_pred['true'].astype(int), window_true_pred['score'])
            
            print('Setting: {}, out: {}, len_window: {}, test_f1: {}, test_precision: {}, test_recall: {},test_accuracy: {}'.format(setting, out, len(window_true_pred), test_f1, test_precision, test_recall, test_accuracy))
            test_f1_fold.append(test_f1)
            test_acc_fold.append(test_accuracy)
            test_precision_fold.append(test_precision)
            test_recall_fold.append(test_recall)
            test_jaccard_fold.append(test_jaccard)
            test_auc_roc_fold.append(test_auc_roc)

        print("Setting: ", setting)
        print('test_f1_fold_mean: {:.6f}, test_f1_fold_std: {:.6f}'.format(np.mean(test_f1_fold), np.std(test_f1_fold)))
        print('test_accuracy_fold_mean: {:.6f}, test_accuracy_fold_std: {:.6f}'.format(np.mean(test_acc_fold), np.std(test_acc_fold)))
        print('test_precision_fold_mean: {:.6f}, test_precision_fold_std: {:.6f}'.format(np.mean(test_precision_fold), np.std(test_precision_fold)))
        print('test_recall_fold_mean: {:.6f}, test_recall_fold_std: {:.6f}'.format(np.mean(test_recall_fold), np.std(test_recall_fold)))
        print('test_jacccard_fold_mean: {:.6f}, test_jaccard_fold_std: {:.6f}'.format(np.mean(test_jaccard_fold), np.std(test_jaccard_fold)))
        print('test_auc_roc_fold_mean: {:.6f}, test_auc_roc_fold_std: {:.6f}'.format(np.mean(test_auc_roc_fold), np.std(test_auc_roc_fold)))
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='feature extraction')
    parser.add_argument('-exp', default='resnet-50-shuffle-roc_auc-2channel', type=str, help='exp name')
    parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
    parser.add_argument('-gpu_id', type=str, nargs='?',default="cuda:0", help="device id to run")
    parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
    parser.add_argument('-t', '--train', default=64, type=int, help='train batch size, default 400')
    parser.add_argument('-v', '--val', default=64, type=int, help='valid batch size, default 10')
    parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 4')
    parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
    parser.add_argument('-c', '--crop', default=0, type=int, help='0 no, 1 cent, 2 rand, 5 five_crop, 10 ten_crop, default 0')
    parser.add_argument('-sbt', '--save_best_type', default=0, type=int, help='0 for auc, 1 for f1, default 0')
    parser.add_argument('-l', '--lr', default=1e-4, type=float, help='learning rate for optimizer, default 1e-4')
    parser.add_argument('-th', '--threshold', default=0.5, type=float, help='threshold for error or normal, default 0.5')


    args = parser.parse_args()

    logger = CompleteLogger("./exp_log/{}_{}/{}".format(args.lr, args.train, args.exp+'generate_window_pred_result'))


    train_batch_size = args.train
    val_batch_size = args.val
    workers = args.work
    threshold = args.threshold

    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    print('experiment name : {}'.format(args.exp))
    print('train batch size: {:6d}'.format(train_batch_size))
    print('threshold: {}'.format(threshold))
    main()

    print('Done')

    logger.close()
    