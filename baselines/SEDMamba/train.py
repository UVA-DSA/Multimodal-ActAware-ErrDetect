import os
import csv
import copy
import random
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from logger import CompleteLogger
from dataload_rarp import CustomVideoDataset
from eval_utils import (
    compute_binary_metrics,
    compute_window_binary_metrics,
    format_percentage,
    make_worker_init_fn,
    metric_key,
    save_video_outputs,
)
from baseline.SEDMamba import MultiStageModel

# Train the model
def train_model(args, data_split_train_path, data_split_test_path):
    criterion = nn.BCEWithLogitsLoss().to(device)
    model = MultiStageModel(args.num_block, args.com_factor, args.features_dim, args.num_class)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_dataset = CustomVideoDataset(data_split_train_path)
    test_dataset = CustomVideoDataset(data_split_test_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.work,
        worker_init_fn=make_worker_init_fn(args.seed),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.work,
        worker_init_fn=make_worker_init_fn(args.seed),
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_metrics = None
    best_test_window_metrics = None
    best_test_window_count = 0
    best_epoch = 0
    best_outputs = None

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0.0
        train_all_scores = []
        train_all_preds = []
        train_all_labels = []

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            video_fe, vl, e_labels = data[0].to(device), data[1], data[2].squeeze(0).to(device)
            video_fe = video_fe.transpose(2, 1)

            predictions = model.forward(video_fe).squeeze(0).squeeze(0)
            loss = criterion(predictions, e_labels.float())
            scores = torch.sigmoid(predictions)
            preds = torch.round(scores)

            loss.backward()
            optimizer.step()

            train_all_scores.extend(scores.flatten().tolist())
            train_all_preds.extend(preds.flatten().tolist())
            train_all_labels.extend(e_labels.flatten().tolist())

            train_loss += loss.data.item()

        train_average_loss = float(train_loss) / len(train_dataloader)
        train_metrics = compute_binary_metrics(train_all_labels, train_all_scores, train_all_preds)

        model.eval()
        test_loss = 0.0
        test_all_scores = []
        test_all_preds = []
        test_all_labels = []
        test_each_vidoe_names = []
        test_video_lengths = []

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                video_fe, vl, e_labels, video_name = data[0].to(device), data[1], data[2].squeeze(0).to(device), data[3]
                video_fe = video_fe.transpose(2, 1)

                predictions = model.forward(video_fe).squeeze(0).squeeze(0)
                loss = criterion(predictions, e_labels.float())
                test_scores = torch.sigmoid(predictions)
                test_preds = torch.round(test_scores)

                test_all_scores.extend(test_scores.flatten().tolist())
                test_all_preds.extend(test_preds.flatten().tolist())
                test_all_labels.extend(e_labels.flatten().tolist())
                test_loss += loss.data.item()

                test_each_vidoe_names.append(video_name[0])
                test_video_lengths.append(int(vl.data[0]))

        test_average_loss = float(test_loss) / len(test_dataloader)
        test_metrics = compute_binary_metrics(test_all_labels, test_all_scores, test_all_preds)
        test_window_metrics, test_window_count = compute_window_binary_metrics(
            test_all_scores,
            test_all_labels,
            test_video_lengths,
            window_length=args.test_window_length,
            window_stride=args.test_window_stride,
        )

        print(
            "epoch: {}"
            " train loss: {:4.4f}"
            " train AUC: {}"
            " train mAP: {}"
            " train F1: {}"
            " train Acc: {}"
            " train Jaccard: {}"
            " test loss: {:4.4f}"
            " test AUC: {}"
            " test mAP: {}"
            " test F1: {}"
            " test Acc: {}"
            " test Jaccard: {}".format(
                epoch,
                train_average_loss,
                format_percentage(train_metrics["roc_auc"]),
                format_percentage(train_metrics["mAP"]),
                format_percentage(train_metrics["f1"]),
                format_percentage(train_metrics["accuracy"]),
                format_percentage(train_metrics["jaccard"]),
                test_average_loss,
                format_percentage(test_metrics["roc_auc"]),
                format_percentage(test_metrics["mAP"]),
                format_percentage(test_metrics["f1"]),
                format_percentage(test_metrics["accuracy"]),
                format_percentage(test_metrics["jaccard"]),
            )
        )
        print(
            "test-window n: {} len: {} stride: {}"
            " AUC: {}"
            " mAP: {}"
            " F1: {}"
            " Acc: {}"
            " Jaccard: {}".format(
                test_window_count,
                args.test_window_length,
                args.test_window_stride,
                format_percentage(test_window_metrics["roc_auc"]),
                format_percentage(test_window_metrics["mAP"]),
                format_percentage(test_window_metrics["f1"]),
                format_percentage(test_window_metrics["accuracy"]),
                format_percentage(test_window_metrics["jaccard"]),
            )
        )

        current_auc_key = metric_key(test_metrics["roc_auc"])
        current_map_key = metric_key(test_metrics["mAP"])
        best_auc_key = metric_key(best_test_metrics["roc_auc"]) if best_test_metrics is not None else float("-inf")
        best_map_key = metric_key(best_test_metrics["mAP"]) if best_test_metrics is not None else float("-inf")

        if current_auc_key > best_auc_key or (current_auc_key == best_auc_key and current_map_key > best_map_key):
            best_test_metrics = dict(test_metrics)
            best_test_window_metrics = dict(test_window_metrics)
            best_test_window_count = int(test_window_count)
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_outputs = {
                "preds": list(test_all_preds),
                "scores": list(test_all_scores),
                "labels": list(test_all_labels),
                "video_names": list(test_each_vidoe_names),
                "video_lengths": list(test_video_lengths),
            }
            base_name = str(args.exp) + "_best"
            if not os.path.exists("./exp_log/{}/{}/".format(args.lr, args.exp)):
                os.makedirs("./exp_log/{}/{}/".format(args.lr, args.exp))
            torch.save(
                best_model_wts,
                "./exp_log/{}/{}/".format(args.lr, args.exp) + base_name + ".pth",
            )
            print("updated best model: {}, AUC: {}".format(best_epoch, best_test_metrics["roc_auc"]))

    print("best_epoch", str(best_epoch))

    return best_test_metrics, best_test_window_metrics, best_test_window_count, best_outputs

# Main function
def main(args):
    root_data_path = args.data_path
    data_split_train_path = root_data_path + "/train_emb_DINOv2/"
    data_split_test_path = root_data_path + "/test_emb_DINOv2/"

    best_test_metrics, best_test_window_metrics, best_test_window_count, best_outputs = train_model(
        args,
        data_split_train_path,
        data_split_test_path,
    )
    output_dir = "./exp_log/{}/{}/".format(args.lr, args.exp)

    if best_outputs is not None:
        save_video_outputs(
            output_dir,
            best_outputs["video_names"],
            best_outputs["video_lengths"],
            best_outputs["preds"],
            best_outputs["scores"],
            best_outputs["labels"],
        )

    if best_test_metrics is not None:
        print("best_test_mAP: {}".format(format_percentage(best_test_metrics["mAP"])))
        print("best_test_AUC: {}".format(format_percentage(best_test_metrics["roc_auc"])))
        print("best_test_F1: {}".format(format_percentage(best_test_metrics["f1"])))
        print("best_test_Acc: {}".format(format_percentage(best_test_metrics["accuracy"])))
        print("best_test_Jaccard: {}".format(format_percentage(best_test_metrics["jaccard"])))
    if best_test_window_metrics is not None:
        print(
            "best_test_window_count: {} (len={}, stride={})".format(
                best_test_window_count,
                args.test_window_length,
                args.test_window_stride,
            )
        )
        print("best_test_window_mAP: {}".format(format_percentage(best_test_window_metrics["mAP"])))
        print("best_test_window_AUC: {}".format(format_percentage(best_test_window_metrics["roc_auc"])))
        print("best_test_window_F1: {}".format(format_percentage(best_test_window_metrics["f1"])))
        print("best_test_window_Acc: {}".format(format_percentage(best_test_window_metrics["accuracy"])))
        print("best_test_window_Jaccard: {}".format(format_percentage(best_test_window_metrics["jaccard"])))

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED")
    parser.add_argument("-exp", default="SEDMamba", type=str, help="exp name")
    parser.add_argument("-dp", "--data_path", default="/path/to/your/data", type=str, help="path to data")
    parser.add_argument("-gpu_id", type=str, nargs="?", default="cuda:0", help="device id to run")
    parser.add_argument("-w", "--work", default=4, type=int, help="num of workers to use")
    parser.add_argument("-s", "--seed", default=2, type=int, help="random seed")
    parser.add_argument("-e", "--epoch", default=200, type=int, help="epochs to train and val")
    parser.add_argument("-l", "--lr", default=1e-4, type=float, help="learning rate for optimizer")
    
    parser.add_argument("-cls", "--num_class", default=1, type=int, help="num of classes")
    parser.add_argument("-fd", "--features_dim", default=1000, type=int, help="DINOv2 features dim")
    parser.add_argument("-nb", "--num_block", default=3, type=int, help="num of BMSS blocks")
    parser.add_argument("-g", "--com_factor", default=64, type=int, help="compression factor G")
    parser.add_argument("--test_window_length", default=10, type=int, help="window length for additional test-set window metrics")
    parser.add_argument("--test_window_stride", default=6, type=int, help="window stride for additional test-set window metrics")

    args = parser.parse_args()

    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

    print("experiment name : {}".format(args.exp))
    print("num of epochs   : {:6d}".format(args.epoch))
    print("num of workers  : {:6d}".format(args.work))
    print("learning rate   : {:4f}".format(args.lr))
    print("device          : {}".format(device))
    print("seed            : {}".format(args.seed))
    print("test win len    : {:6d}".format(args.test_window_length))
    print("test win stride : {:6d}".format(args.test_window_stride))

    # Initialize seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    logger = CompleteLogger("./exp_log/{}/{}".format(args.lr, args.exp))
    main(args)

    print("Done")
    logger.close()
