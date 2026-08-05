import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score

# Function to calculate AUROC, F1, Precision, Recall, and accuracy scores and plot AUC-ROC curve for binary classification
def binary_evaluation(y_true, y_scores, preds_phase):
    roc_auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    accuracy = accuracy_score(y_true, preds_phase)
    f1 = f1_score(y_true, preds_phase, average='binary', pos_label=1)
    precision = precision_score(y_true, preds_phase, average='binary', pos_label=1)
    recall = recall_score(y_true, preds_phase, average='binary', pos_label=1)
    jaccard = jaccard_score(y_true, preds_phase, average='binary', pos_label=1)
    # f1_macro = f1_score(y_true, preds_phase, average='macro')
    # precision_macro = precision_score(y_true, preds_phase, average='macro')
    # recall_macro = recall_score(y_true, preds_phase, average='macro')
    # jaccard_macro = jaccard_score(y_true, preds_phase, average='macro')
    precision_each = precision_score(y_true, preds_phase, average=None)
    recall_each = recall_score(y_true, preds_phase, average=None)
    class_report = classification_report(y_true, preds_phase, labels=[0,1], digits=6, output_dict=False, zero_division='warn')
    cm = confusion_matrix(y_true, preds_phase)
    # edit, f1_k10, f1_k25, f1_k50 = edit_f1(y_true, preds_phase)

    return roc_auc, cm, f1, jaccard, accuracy, \
        precision, recall, precision_each, recall_each, class_report, \
        fpr, tpr

# Function to calculate AUROC, F1, Precision, Recall, and accuracy scores and plot AUC-ROC curve for multi-class classification
def multiclass_evaluation(y_true, y_scores, preds_phase):
    #y_true = y_true.numpy()
    #y_scores = y_scores.detach().numpy()
    n_classes = y_scores.shape[1]

    roc_auc = []
    f1 = []
    precision = []
    recall = []
    accuracy = []

    for i in range(n_classes):
        roc_auc.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        f1.append(f1_score(y_true[:, i], np.round(y_scores[:, i])))
        precision.append(precision_score(y_true[:, i], np.round(y_scores[:, i])))
        recall.append(recall_score(y_true[:, i], np.round(y_scores[:, i])))
        accuracy.append(accuracy_score(y_true[:, i], np.round(y_scores[:, i])))

    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_scores[:, i])
        plt.plot(fpr, tpr, color=colors[i], label='Class %d (AUC = %0.2f)' % (i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve (Multi-class Classification)')
    plt.legend(loc='lower right')
    plt.show()

    return roc_auc, f1, precision, recall, accuracy, fpr, tpr

#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=0):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] != bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] != bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label != bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label != bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=0):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=0):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def edit_f1(gt_content_all, recog_content_all):
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', default="gtea")
    # parser.add_argument('--split', default='1')

    # args = parser.parse_args()

    # ground_truth_path = "./data/"+args.dataset+"/groundTruth/"
    # recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    # file_list = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"

    # list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp_all, fp_all, fn_all, f1_all =  np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    edit_all = 0

    # for vid in list_of_videos:
    #     gt_file = ground_truth_path + vid
    #     gt_content = read_file(gt_file).split('\n')[0:-1]

    #     recog_file = recog_path + vid.split('.')[0]
    #     recog_content = read_file(recog_file).split('\n')[1].split()

    #     for i in range(len(gt_content)):
    #         total += 1
    #         if gt_content[i] == recog_content[i]:
    #             correct += 1

    #     edit += edit_score(recog_content, gt_content)

    #     for s in range(len(overlap)):
    #         tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
    #         tp[s] += tp1
    #         fp[s] += fp1
    #         fn[s] += fn1
    # print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    # edit = ((1.0*edit)/len(list_of_videos))
    # for s in range(len(overlap)):
    #     precision = tp[s] / float(tp[s]+fp[s])
    #     recall = tp[s] / float(tp[s]+fn[s])

    #     f1 = 2.0 * (precision*recall) / (precision+recall)

    #     f1 = np.nan_to_num(f1)*100
    #     print('F1@%0.2f: %.4f' % (overlap[s], f1))

    edit_all = edit_score(recog_content_all, gt_content_all)
    for s in range(len(overlap)):
        tp1_all, fp1_all, fn1_all = f_score(recog_content_all, gt_content_all, overlap=[s])
        tp_all[s] += tp1_all
        fp_all[s] += fp1_all
        fn_all[s] += fn1_all
        precision_all = tp_all[s] / float(tp_all[s]+fp_all[s])
        recall_all = tp_all[s] / float(tp_all[s]+fn_all[s])
        f1_all[s] = 2.0 * (precision_all*recall_all) / (precision_all+recall_all)
        f1_all[s] = np.nan_to_num(f1_all[s])*100
        print('F1_all@%0.2f: %.4f' % (overlap[s], f1_all[s]))
    
    return edit_all, f1_all[0], f1_all[1], f1_all[2]


if __name__ == '__main__':
    # Example usage for binary classification
    y_true_binary = torch.tensor([0, 1, 1, 0, 1])
    y_scores_binary = torch.tensor([0.1, 0.8, 0.6, 0.3, 0.9])
    binary_auc, binary_f1, binary_precision, binary_recall, binary_accuracy = binary_evaluation(y_true_binary, y_scores_binary)
    print("Binary AUROC Score:", binary_auc)
    print("Binary F1 Score:", binary_f1)
    print("Binary Precision:", binary_precision)
    print("Binary Recall:", binary_recall)
    print("Binary Accuracy:", binary_accuracy)

    # Example usage for multi-class classification
    y_true_multiclass = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
    y_scores_multiclass = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.5, 0.3]])
    multiclass_auc, multiclass_f1, multiclass_precision, multiclass_recall, multiclass_accuracy = multiclass_evaluation(y_true_multiclass, y_scores_multiclass)
    print("Multi-class AUROC Scores:", multiclass_auc)
    print("Multi-class F1 Scores:", multiclass_f1)
    print("Multi-class Precision:", multiclass_precision)
    print("Multi-class Recall:", multiclass_recall)
    print("Multi-class Accuracy:", multiclass_accuracy)
