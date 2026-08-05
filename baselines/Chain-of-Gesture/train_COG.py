import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import numpy as np
import models
import pickle, time
import random
from sklearn import metrics
import copy
import os
import argparse
from logger import CompleteLogger

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, jaccard_score
from metrics import binary_evaluation
from matplotlib import pyplot as plt
import csv
from dataload import CustomVideoDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pdb
# import warnings
# warnings.filterwarnings("ignore")

# Initialize seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fn(num_workers, rank, seed):
    worker_seed = num_workers * rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def train_model(args, setting, out, data_split_path):
    torch.cuda.empty_cache()
    summary_writer = SummaryWriter("./exp_log/{}_{}/{}/{}_{}/run".format(args.lr, args.train, args.exp, setting, out))
    out_features = 2 # num_class
    mstcn_causal_conv = True
    mstcn_layers = args.layers#10#8
    mstcn_f_maps = 64#64
    mstcn_f_dim= 2048
    mstcn_stages = args.stages
    fast_frames = args.train
    num_R = 3
    num_layers_R = args.layers#10
    num_layers_Basic = 11

    d_model = args.dmodel #128#64
    d_q = int(d_model/8)#16#8
    len_q = args.len_q#16


    criterion= nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

 
    model = models.COG(args, num_layers_Basic, num_layers_R, num_R, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv, d_model, d_q, len_q, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #ignored_params = list(map(id, model.cot.parameters()))
    #base_params = filter(lambda p:id(p) not in ignored_params, model.parameters())
    #optimizer = optim.Adam([{'params': base_params}, {'params': model.cot.parameters(), 'lr': 1e-3}], lr=args.lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_roc_auc = 0.0
    best_test_f1 = 0.0
    best_test_jaccard = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    train_dataset = CustomVideoDataset(data_split_path, train=True)
    test_dataset = CustomVideoDataset(data_split_path, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=workers, worker_init_fn=worker_init_fn(workers, 0, seed))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers, worker_init_fn=worker_init_fn(workers, 0, seed))

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        train_start_time = time.time()

        train_all_scores = []
        train_all_preds = []
        train_all_labels = []
        batch_progress = 0
        train_batch_size = 1
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            video_fe, vl, e_labels, g_labels = data[0].to(device), data[1], data[2].squeeze(0).to(device), data[3].squeeze(0).to(device)
            #video_fe = video_fe.transpose(2, 1) #[1, T, Features] --> [1, Features, T]
            predicted_list, feature_list = model.forward(video_fe)
            all_out, resize_list, labels_list = models.fusion(predicted_list, e_labels)
            clc_loss = 0.0
            smooth_loss = 0.0
            for p, l in zip(resize_list, labels_list):
                p_classes = p.squeeze(0).transpose(1,0)
                ce_loss = criterion(p_classes.squeeze(), l)
                sm_loss = torch.mean(torch.clamp(criterion2(F.log_softmax(p_classes[1:, :], dim=1), F.log_softmax(p_classes.detach()[:-1, :], dim=1)), min=0, max=16))
                clc_loss += ce_loss
                smooth_loss += sm_loss

            clc_loss = clc_loss / (mstcn_stages * 1.0)
            smooth_loss = smooth_loss / (mstcn_stages * 1.0)

            _, preds = torch.max(resize_list[0].squeeze().transpose(1, 0).data, 1)
            train_p_classes = torch.softmax((resize_list[0].squeeze().transpose(1, 0)), dim=1)
            train_p_classes_positive = train_p_classes[:, 1]

            loss = clc_loss + args.lambda_ * smooth_loss
            loss.backward()
            optimizer.step()

            for j in range(len(train_p_classes_positive)):
                train_all_scores.append(float(train_p_classes_positive.data[j]))
            for index in range(len(preds)):
                train_all_preds.append(int(preds.data[index]))
            for index in range(len(e_labels)):
                train_all_labels.append(int(e_labels.data[index]))

            train_loss += loss.data.item()
            iteration = epoch *len(train_dataloader) + i

            summary_writer.add_scalar('train/video_loss', loss, iteration)

            batch_progress += 1
            if batch_progress * train_batch_size >= len(train_dataset):
                percent = 100.0
                print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress * train_batch_size, len(train_dataset)), end='\n')
            else:
                percent = round(batch_progress * train_batch_size / len(train_dataset) * 100, 2)
                print('Batch progress: %s [%d/%d]' % (
                str(percent) + '%', batch_progress * train_batch_size, len(train_dataset)), end='\r')

        train_elapsed_time = time.time() - train_start_time
        train_average_loss = float(train_loss) / len(train_dataloader)
        train_jaccard = jaccard_score(train_all_labels, train_all_preds, average='binary', pos_label=1)
        train_auc_roc = roc_auc_score(train_all_labels, train_all_scores)
        train_f1 = f1_score(train_all_labels, train_all_preds, average='binary', pos_label=1)
        train_accuracy = accuracy_score(train_all_labels, train_all_preds)


        # Sets the module in evaluation mode.
        model.eval()
        test_all_scores = []
        test_all_preds = []
        test_all_labels = []
        test_all_gest_labels = []
        test_each_video_names = []
        test_each_video_lengths = []
        test_start_time = time.time()
        test_progress = 0
        val_batch_size = 1
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                video_fe, vl, e_labels, g_labels, video_name  = data[0].to(device), data[1], data[2].squeeze(0).to(device), data[3].squeeze(0), data[4]
                #video_fe = video_fe.transpose(2, 1) #[1, 2048, bs]
                
                predicted_list, feature_list = model.forward(video_fe)
                all_out, resize_list, labels_list = models.fusion(predicted_list, e_labels)
                test_clc_loss = 0.0
                test_smooth_loss = 0.0
                for p, l in zip(resize_list, labels_list):
                    p_classes = p.squeeze(0).transpose(1,0)

                    ce_loss = criterion(p_classes.squeeze(), l)
                    sm_loss = torch.mean(torch.clamp(criterion2(F.log_softmax(p_classes[1:, :], dim=1), F.log_softmax(p_classes.detach()[:-1, :], dim=1)), min=0, max=16))
                    test_clc_loss += ce_loss
                    test_smooth_loss += sm_loss

                test_clc_loss = test_clc_loss / (mstcn_stages * 1.0)
                test_smooth_loss = test_smooth_loss / (mstcn_stages * 1.0)

               

                test_loss += test_clc_loss + args.lambda_ * test_smooth_loss

                _, preds = torch.max(predicted_list[0].squeeze().transpose(1, 0).data, 1)
                p_classes = torch.softmax((predicted_list[0].squeeze().transpose(1, 0)), dim=1)
                p_classes_positive = p_classes[:, 1]

                for j in range(len(p_classes_positive)):
                    test_all_scores.append(float(p_classes_positive.data[j]))
                for j in range(len(preds)):
                    test_all_preds.append(int(preds.data[j]))
                for j in range(len(e_labels)):
                    test_all_labels.append(int(e_labels.data[j]))
                for j in range(len(g_labels)):
                    test_all_gest_labels.append(int(g_labels.data[j]))
                
                test_each_video_names.append(video_name[0])
                test_each_video_lengths.append(int(vl.data[0]))
                test_progress += 1
                if test_progress * val_batch_size >= len(test_dataset):
                    percent = 100.0
                    print('test progress: %s [%d/%d]' % (str(percent) + '%', test_progress * val_batch_size, len(test_dataset)), end='\n')
            # else:
            #     percent = round(test_progress * val_batch_size / len(test_dataset) * 100, 2)
            #     print('test progress: %s [%d/%d]' % (str(percent) + '%', test_progress * val_batch_size, len(test_dataset)),
            #           end='\r')
                
        test_elapsed_time = time.time() - test_start_time
        test_roc_auc, test_cm, test_f1, test_jaccard, test_accuracy, \
        test_precision, test_recall, test_precision_each, test_recall_each, test_class_report, \
        test_fpr, test_tpr \
        = binary_evaluation(test_all_labels, test_all_scores, test_all_preds)
        summary_writer.add_scalar('train/loss', train_loss, epoch)
        summary_writer.add_scalar('train/f1', train_f1, epoch)
        summary_writer.add_scalar('train/acc', train_accuracy, epoch)
        summary_writer.add_scalar('test/loss', test_loss, epoch)
        summary_writer.add_scalar('test/f1', test_f1, epoch)
        summary_writer.add_scalar('test/acc', test_accuracy, epoch)

        print('epoch: {}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss: {:4.4f}'
              ' train auc: {:4f}'
              ' train accu: {:4f}'
              ' train f1: {:4f}'
              ' train jaccard: {:4f}'
              ' test in: {:2.0f}m{:2.0f}s'
              ' test auc: {:4f}'
              ' test accu: {:4f}'
              ' test f1: {:4f}'
              ' test jaccard: {:4f}'             
              ' test recall: {:4f}'
              ' test precision: {:4f}'
              ' test precision_each: {}'
              ' test recall_each: {}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_average_loss,
                      train_auc_roc,
                      train_accuracy,
                      train_f1,
                      train_jaccard,
                      test_elapsed_time // 60,
                      test_elapsed_time % 60,
                      test_roc_auc,
                      test_accuracy,
                      test_f1,
                      test_jaccard,
                      test_recall,
                      test_precision,
                      test_precision_each,
                      test_recall_each
                      ))
        print('test_class_report:', test_class_report)
        print('test_cm:', test_cm)
        if args.save_best_type == 0:
            if test_roc_auc > best_test_roc_auc or (test_roc_auc == best_test_roc_auc and test_jaccard > best_test_jaccard) \
                or (test_roc_auc == best_test_roc_auc and test_jaccard == best_test_jaccard and test_f1 > best_test_f1) \
                    or (test_roc_auc == best_test_roc_auc and test_jaccard == best_test_jaccard and test_f1 == best_test_f1 and test_accuracy > best_test_acc):
                best_test_roc_auc = test_roc_auc
                best_test_f1 = test_f1
                best_test_jaccard = test_jaccard
                best_test_acc = test_accuracy
                correspond_test_precision = test_precision
                correspond_test_recall = test_recall
                correspond_test_fpr = test_fpr
                correspond_test_tpr = test_tpr
                correspond_test_video_preds = test_all_preds
                correspond_test_video_labels = test_all_labels
                correspond_test_video_gest_labels = test_all_gest_labels
                correspond_test_video_names = test_each_video_names
                correspond_test_video_lengths = test_each_video_lengths
                correspond_train_auc_roc = train_auc_roc
                correspond_train_f1 = train_f1
                correspond_train_acc = train_accuracy
                correspond_train_jaccard = train_jaccard
                correspond_test_class_report = test_class_report
                correspond_test_cm = test_cm
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                base_name = str(args.exp) \
                        + "_" + setting \
                        + "_" + out \
                        + "_best"
                if not os.path.exists("./exp_log/{}_{}/{}/{}_{}".format(args.lr, args.train, args.exp, setting, out)):
                    os.makedirs("./exp_log/{}_{}/{}/{}_{}".format(args.lr, args.train, args.exp, setting, out))
                torch.save(best_model_wts, "./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out) + base_name + ".pth")
                print('updated best model: {}, auc: {}'.format(best_epoch, best_test_roc_auc))
        elif args.save_best_type == 1:
            if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_jaccard > best_test_jaccard) \
                or (test_f1 == best_test_f1 and test_jaccard == best_test_jaccard and test_roc_auc > best_test_roc_auc) \
                    or (test_f1 == best_test_f1 and test_jaccard == best_test_jaccard and test_roc_auc == best_test_roc_auc and test_accuracy > best_test_acc):
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_jaccard = test_jaccard
                best_test_acc = test_accuracy
                correspond_test_precision = test_precision
                correspond_test_recall = test_recall
                correspond_test_fpr = test_fpr
                correspond_test_tpr = test_tpr
                correspond_test_video_preds = test_all_preds
                correspond_test_video_labels = test_all_labels
                correspond_test_video_gest_labels = test_all_gest_labels
                correspond_test_video_names = test_each_video_names
                correspond_test_video_lengths = test_each_video_lengths
                correspond_train_auc_roc = train_auc_roc
                correspond_train_f1 = train_f1
                correspond_train_acc = train_accuracy
                correspond_train_jaccard = train_jaccard
                correspond_test_class_report = test_class_report
                correspond_test_cm = test_cm
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                base_name = str(args.exp) \
                        + "_" + setting \
                        + "_" + out \
                        + "_best"
                if not os.path.exists("./exp_log/{}_{}/{}/{}_{}".format(args.lr, args.train, args.exp, setting, out)):
                    os.makedirs("./exp_log/{}_{}/{}/{}_{}".format(args.lr, args.train, args.exp, setting, out))
                torch.save(best_model_wts, "./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out) + base_name + ".pth")
                print('updated best model: {}, f1: {} auc: {}'.format(best_epoch, best_test_f1, best_test_roc_auc))
        elif args.save_best_type == 2:
            if test_jaccard > best_test_jaccard or (test_jaccard == best_test_jaccard and test_f1 > best_test_f1) \
                or (test_jaccard == best_test_jaccard and test_f1 == best_test_f1 and test_roc_auc > best_test_roc_auc) \
                    or (test_jaccard == best_test_jaccard and test_f1 == best_test_f1 and test_roc_auc == best_test_roc_auc and test_accuracy > best_test_acc):
                best_test_jaccard = test_jaccard
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_acc = test_accuracy
                correspond_test_precision = test_precision
                correspond_test_recall = test_recall
                correspond_test_fpr = test_fpr
                correspond_test_tpr = test_tpr
                correspond_test_video_preds = test_all_preds
                correspond_test_video_labels = test_all_labels
                correspond_test_video_gest_labels = test_all_gest_labels
                correspond_test_video_names = test_each_video_names
                correspond_test_video_lengths = test_each_video_lengths
                correspond_train_auc_roc = train_auc_roc
                correspond_train_f1 = train_f1
                correspond_train_acc = train_accuracy
                correspond_train_jaccard = train_jaccard
                correspond_test_class_report = test_class_report
                correspond_test_cm = test_cm
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                base_name = str(args.exp) \
                        + "_" + setting \
                        + "_" + out \
                        + "_best"
                if not os.path.exists("./exp_log/{}_{}/{}/{}_{}".format(args.lr, args.train, args.exp, setting, out)):
                    os.makedirs("./exp_log/{}_{}/{}/{}_{}".format(args.lr, args.train, args.exp, setting, out))
                torch.save(best_model_wts, "./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out) + base_name + ".pth")
                print('updated best model: {}, jaccard: {} f1: {} auc: {}'.format(best_epoch, best_test_jaccard, best_test_f1, best_test_roc_auc))

    print("Setting:", setting, out, "best_epoch", str(best_epoch))

    return best_test_roc_auc, best_test_f1, best_test_jaccard, \
        best_test_acc, correspond_test_precision, correspond_test_recall, \
        correspond_test_fpr, correspond_test_tpr, correspond_test_video_preds, correspond_test_video_labels, correspond_test_video_gest_labels, correspond_test_video_names, correspond_test_video_lengths, \
        correspond_train_f1, correspond_train_acc, correspond_train_auc_roc, correspond_train_jaccard, correspond_test_class_report, correspond_test_cm
        
def main():
    settings = ['LOSO']#, 'LOUO']
    tasks = ['All', 'Suturing','Knot_Tying','Needle_Passing']
    out1 = ['Bout','Cout','Dout','Eout','Fout','Gout','Hout','Iout']
    out2 = ['1out','2out','3out','4out','5out']

    for setting in settings:
        test_roc_auc_fold = []
        test_f1_fold = []
        test_jaccard_fold = []
        test_acc_fold = []
        test_precision_fold = []
        test_recall_fold = []
        test_fpr_fold = []
        test_tpr_fold = []
        test_class_report_fold = []
        test_cm_fold = []
        train_f1_fold = []
        train_acc_fold = []
        train_auc_roc_fold = []
        train_jaccard_fold = []

        if setting == 'LOSO':
            outs = out2
        else:
            outs = out1

        for out in outs:
            root_data_path = './dataset'
            if args.save_best_type == 0:
                data_split_path = root_data_path + '/setting/' + setting + '/All/' + out
            elif args.save_best_type == 1:
                data_split_path = root_data_path + '/setting_f1/' + setting + '/All/' + out
            elif args.save_best_type == 2:
                data_split_path = root_data_path + '/setting_jc/' + setting + '/All/' + out
            best_test_roc_auc, best_test_f1, best_test_jaccard, \
            best_test_acc, correspond_test_precision, correspond_test_recall, \
            correspond_test_fpr, correspond_test_tpr, correspond_test_video_preds,correspond_test_video_labels, correspond_test_video_gest_labels, correspond_test_video_names, correspond_test_video_lengths, \
            correspond_train_f1, correspond_train_acc, correspond_train_auc_roc, correspond_train_jaccard, correspond_test_class_report, correspond_test_cm \
            = train_model(args, setting, out, data_split_path)
           
            result_filename = "./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out) + "results.csv"
            with open(result_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                n = len(correspond_test_video_gest_labels)
                for i in range(n):
                    csvwriter.writerow([correspond_test_video_gest_labels[i],correspond_test_video_labels[i], correspond_test_video_preds[i]])

            start_idx = 0
            for i in range(len(correspond_test_video_names)):
                preds_filename = "./exp_log/{}_{}/{}/{}_{}/".format(args.lr, args.train, args.exp, setting, out) + correspond_test_video_names[i]
                with open(preds_filename, 'w') as f:
                    writer = csv.writer(f)
                    for j in range(correspond_test_video_lengths[i]):
                        writer.writerow([correspond_test_video_gest_labels[start_idx + j],correspond_test_video_labels[start_idx + j], correspond_test_video_preds[start_idx + j]])
                start_idx += correspond_test_video_lengths[i]


            test_roc_auc_fold.append(best_test_roc_auc)
            test_f1_fold.append(best_test_f1)
            test_jaccard_fold.append(best_test_jaccard)
            test_acc_fold.append(best_test_acc)
            test_precision_fold.append(correspond_test_precision)
            test_recall_fold.append(correspond_test_recall)
            test_fpr_fold.append(correspond_test_fpr)
            test_tpr_fold.append(correspond_test_tpr)
            test_class_report_fold.append(correspond_test_class_report)
            test_cm_fold.append(correspond_test_cm)

            train_f1_fold.append(correspond_train_f1)
            train_acc_fold.append(correspond_train_acc)
            train_auc_roc_fold.append(correspond_train_auc_roc)
            train_jaccard_fold.append(correspond_train_jaccard)

            
        print('Setting:', setting)
        print('train_f1_fold:', train_f1_fold)
        print('train_acc_fold:', train_acc_fold)
        print('train_auc_roc_fold:', train_auc_roc_fold)
        print('train_jaccard_fold:', train_jaccard_fold)
        print('train_auc_roc_fold_mean: {:.6f}, train_auc_roc_fold_std: {:.6f}'.format(np.mean(train_auc_roc_fold), np.std(train_auc_roc_fold)))
        print('train_f1_fold_mean: {:.6f}, train_f1_fold_std: {:.6f}'.format(np.mean(train_f1_fold), np.std(train_f1_fold)))
        print('train_acc_fold_mean: {:.6f}, train_acc_fold_std: {:.6f}'.format(np.mean(train_acc_fold), np.std(train_acc_fold)))
        print('train_jaccard_fold_mean: {:.6f}, train_jaccard_fold_std: {:.6f}'.format(np.mean(train_jaccard_fold), np.std(train_jaccard_fold)))
        print('test_class_report_fold:', test_class_report_fold)
        print('test_cm_fold:', test_cm_fold)
        print('test_roc_auc_fold:', test_roc_auc_fold)
        print('test_f1_fold:', test_f1_fold)
        print('test_jaccard_fold:', test_jaccard_fold)
        print('test_acc_fold:', test_acc_fold)
        print('test_precision_fold:', test_precision_fold)
        print('test_recall_fold:', test_recall_fold)
        print('test_recall_fold_mean: {:.6f}, test_recall_fold_std: {:.6f}'.format(np.mean(test_recall_fold), np.std(test_recall_fold)))        
        print('test_precision_fold_mean: {:.6f}, test_precision_fold_std: {:.6f}'.format(np.mean(test_precision_fold), np.std(test_precision_fold)))
        print('test_jaccard_fold_mean: {:.6f}, test_jaccard_fold_std: {:.6f}'.format(np.mean(test_jaccard_fold), np.std(test_jaccard_fold)))
        print('test_acc_fold_mean: {:.6f}, test_acc_fold_std: {:.6f}'.format(np.mean(test_acc_fold), np.std(test_acc_fold)))
        print('test_f1_fold_mean: {:.6f}, test_f1_fold_std: {:.6f}'.format(np.mean(test_f1_fold), np.std(test_f1_fold)))
        print('test_roc_auc_fold_mean: {:.6f}, test_roc_auc_fold_std: {:.6f}'.format(np.mean(test_roc_auc_fold), np.std(test_roc_auc_fold)))

        plt.figure()
        if setting == 'LOSO':
            colors = ['b', 'g', 'c', 'm', 'y']
            for i in range(len(outs)):
                plt.plot(test_fpr_fold[i], test_tpr_fold[i], color=colors[i], alpha=0.5, label='Fold {} (AUC = {:.2f})'.format(i+1, test_roc_auc_fold[i]))
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Cross-Validation')
            plt.legend(loc='lower right')
            plt.savefig("./exp_log/{}_{}/{}/plot_{}_{}.png".format(args.lr, args.train, args.exp, setting, 'roc'))

            # plt.figure()
            # plt.plot(np.mean(test_fpr_fold, axis=0), np.mean(test_tpr_fold, axis=0), color='k', linestyle='-', linewidth=2, label='Average ROC (AUC = {:.2f})'.format(np.mean(test_roc_auc_fold)))
            # plt.plot([0, 1], [0, 1], 'r--')
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC Curve - Average')
            # plt.legend(loc='lower right')
            # plt.savefig("./exp_log/{}_{}/{}/plot_{}_{}.png".format(args.lr, args.train, args.exp, setting, 'roc_average'))
        else:
            colors = ['b', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']
            for i in range(len(outs)):
                plt.plot(test_fpr_fold[i], test_tpr_fold[i], color=colors[i], alpha=0.5, label='Fold {} (AUC = {:.2f})'.format(i+1, test_roc_auc_fold[i]))
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Cross-Validation')
            plt.legend(loc='lower right')
            plt.savefig("./exp_log/{}_{}/{}/plot_{}_{}.png".format(args.lr, args.train, args.exp, setting, 'roc'))

            # plt.figure()
            # plt.plot(np.mean(test_fpr_fold, axis=0), np.mean(test_tpr_fold, axis=0), color='k', linestyle='-', linewidth=2, label='Average ROC (AUC = {:.2f})'.format(np.mean(test_roc_auc_fold)))
            # plt.plot([0, 1], [0, 1], 'r--')
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC Curve - Average')
            # plt.legend(loc='lower right')
            # plt.savefig("./exp_log/{}_{}/{}/plot_{}_{}.png".format(args.lr, args.train, args.exp, setting, 'roc_average'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COG')
    parser.add_argument('-exp', default='COG', type=str, help='exp name')
    parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
    parser.add_argument('-gpu_id', type=str, nargs='?',default="cuda:0", help="device id to run")
    parser.add_argument('-t', '--train', default=1, type=int, help='train batch size, default 400')
    parser.add_argument('-e', '--epo', default=50, type=int, help='epochs to train and val, default 25')
    parser.add_argument('-w', '--work', default=4, type=int, help='num of workers to use, default 4')
    parser.add_argument('-l', '--lr', default=5e-4, type=float, help='learning rate for optimizer, default 5e-5')
    parser.add_argument('-sbt', '--save_best_type', default=1, type=int, help='0 for auc, 1 for f1, default 0')
    parser.add_argument('-lambda_', default=0.15, type=float, help='coefficient of smooth loss, default 0.15')
    parser.add_argument('-dmodel', default= 64, type=int, help='dimension of transformer')
    parser.add_argument('-len_q', default= 40, type=int, help='length of sequence in transformer')
    parser.add_argument('-k', default=16, type=int, help='stride and kernel size of pooling for downsampling' )
    parser.add_argument('-layers', default=10, type=int, help='layers of mstcn')
    parser.add_argument('-stages', default=8, type=int, help='stages of mstcn' )

    args = parser.parse_args()
    logger = CompleteLogger("./exp_log/{}_{}/{}".format(args.lr, args.train, args.exp))

    epochs = args.epo
    workers = args.work
    learning_rate = args.lr

    # num_gpu = torch.cuda.device_count()
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

    # print('number of gpu   : {:6d}'.format(num_gpu))
    print('experiment name : {}'.format(args.exp))
    print('train batch size: {:6d}'.format(args.train))
    print('num of epochs   : {:6d}'.format(epochs))
    print('num of workers  : {:6d}'.format(workers))
    print('learning rate   : {:4f}'.format(learning_rate))

    main()

    print('Done')

    logger.close()
