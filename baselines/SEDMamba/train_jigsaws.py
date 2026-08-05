import argparse
import copy
import csv
import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from baseline.SEDMamba import MultiStageModel
from dataload_jigsaws import CustomVideoDataset
from eval_utils import (
    compute_binary_metrics,
    compute_window_binary_metrics,
    format_percentage,
    make_worker_init_fn,
    metric_key,
    save_video_outputs,
)
from logger import CompleteLogger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jigsaws_splits import get_split_fold_ids, load_split_records, make_dataset_variant  # noqa: E402


DEFAULT_PKL_ROOT = os.path.join(REPO_ROOT, "data", "jigsaws_sar_rarp_pkls", "resnet50")
DEFAULT_SPLIT_ROOT = os.path.join(REPO_ROOT, "LOSO")
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "exp_log_jigsaws")
DEFAULT_TASKS = ("Suturing", "Needle_Passing")
DEFAULT_TEST_WINDOW_LENGTH = 20
DEFAULT_TEST_WINDOW_STRIDE = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def infer_feature_dim(pkl_root: str, video_names: Sequence[str]) -> int:
    if not video_names:
        raise ValueError("Cannot infer feature dimension without any videos.")
    dataset = CustomVideoDataset(pkl_root, video_names=[video_names[0]])
    features, _, _, _ = dataset[0]
    return int(features.shape[-1])


def collect_fold_records(
    tasks: Sequence[str],
    split_scheme: str,
    fold_id: int,
    split_root: str,
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    train_records: List[str] = []
    test_records: List[str] = []
    task_stats: List[Dict[str, object]] = []

    for task in tasks:
        dataset_variant = make_dataset_variant(task, split_scheme, fold_id)
        split_records = load_split_records(dataset_variant, split_root=split_root, repo_root=REPO_ROOT)
        task_train = list(split_records["train"])
        task_test = list(split_records["test"])
        train_records.extend(task_train)
        test_records.extend(task_test)
        task_stats.append(
            {
                "task": task,
                "train_count": len(task_train),
                "test_count": len(task_test),
            }
        )

    return train_records, test_records, task_stats


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    train_mode: bool,
) -> Dict[str, object]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_scores: List[float] = []
    all_preds: List[float] = []
    all_labels: List[float] = []
    video_names: List[str] = []
    video_lengths: List[int] = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for data in dataloader:
            if train_mode:
                optimizer.zero_grad()

            video_fe = data[0].to(device)
            video_length = int(data[1].data[0])
            error_labels = data[2].squeeze(0).to(device)
            video_name = data[3][0]

            video_fe = video_fe.transpose(2, 1)
            predictions = model.forward(video_fe).squeeze(0).squeeze(0)
            loss = criterion(predictions, error_labels.float())
            scores = torch.sigmoid(predictions)
            preds = torch.round(scores)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            all_scores.extend(scores.flatten().detach().cpu().tolist())
            all_preds.extend(preds.flatten().detach().cpu().tolist())
            all_labels.extend(error_labels.flatten().detach().cpu().tolist())
            video_names.append(video_name)
            video_lengths.append(video_length)

    metrics = compute_binary_metrics(all_labels, all_scores, all_preds)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    metrics["video_names"] = video_names
    metrics["video_lengths"] = video_lengths
    metrics["all_scores"] = all_scores
    metrics["all_preds"] = all_preds
    metrics["all_labels"] = all_labels
    return metrics


def serialize_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keep_keys = [
        "loss",
        "roc_auc",
        "mAP",
        "f1",
        "accuracy",
        "jaccard",
        "num_windows",
        "window_length",
        "window_stride",
        "score_aggregation",
        "label_aggregation",
    ]
    return {key: metrics[key] for key in keep_keys if key in metrics}


def compute_test_window_metrics(
    metrics: Dict[str, object],
    window_length: int,
    window_stride: int,
) -> Dict[str, object]:
    window_metrics, window_count = compute_window_binary_metrics(
        metrics["all_scores"],
        metrics["all_labels"],
        metrics["video_lengths"],
        window_length=window_length,
        window_stride=window_stride,
    )
    window_metrics["num_windows"] = int(window_count)
    window_metrics["window_length"] = int(window_length)
    window_metrics["window_stride"] = int(window_stride)
    window_metrics["score_aggregation"] = "max"
    window_metrics["label_aggregation"] = "any_error"
    return window_metrics


def build_dataloaders(
    args,
    train_records: Sequence[str],
    test_records: Sequence[str],
):
    train_dataset = CustomVideoDataset(args.pkl_root, video_names=train_records)
    test_dataset = CustomVideoDataset(args.pkl_root, video_names=test_records)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.work,
        worker_init_fn=make_worker_init_fn(args.seed),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.work,
        worker_init_fn=make_worker_init_fn(args.seed),
    )
    return train_dataset, test_dataset, train_loader, test_loader


def train_fold(args, fold_id: int, device: torch.device, feature_dim: int) -> Dict[str, object]:
    train_records, test_records, task_stats = collect_fold_records(
        args.tasks,
        args.split_scheme,
        fold_id,
        args.split_root,
    )
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        args,
        train_records,
        test_records,
    )

    model = MultiStageModel(args.num_block, args.com_factor, feature_dim, args.num_class).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    fold_dir = os.path.join(args.run_dir, "{}out".format(fold_id))
    os.makedirs(fold_dir, exist_ok=True)

    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_metrics = None
    best_test_window_metrics = None
    best_outputs = None
    epoch_logs = []

    print("\n=== Fold {}out ===".format(fold_id))
    print("tasks              :", ",".join(args.tasks))
    print("num_train_videos   :", len(train_dataset))
    print("num_test_videos    :", len(test_dataset))
    print(
        "test window eval   : len={} stride={} score=max label=any_error".format(
            args.test_window_length,
            args.test_window_stride,
        )
    )

    for task_stat in task_stats:
        print(
            "  {} -> train={} test={}".format(
                task_stat["task"], task_stat["train_count"], task_stat["test_count"]
            )
        )

    for epoch in range(args.epoch):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        test_metrics = run_epoch(model, test_loader, criterion, None, device, train_mode=False)
        test_window_metrics = compute_test_window_metrics(
            test_metrics,
            window_length=args.test_window_length,
            window_stride=args.test_window_stride,
        )

        print(
            "fold: {} epoch: {}"
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
                fold_id,
                epoch,
                train_metrics["loss"],
                format_percentage(train_metrics["roc_auc"]),
                format_percentage(train_metrics["mAP"]),
                format_percentage(train_metrics["f1"]),
                format_percentage(train_metrics["accuracy"]),
                format_percentage(train_metrics["jaccard"]),
                test_metrics["loss"],
                format_percentage(test_metrics["roc_auc"]),
                format_percentage(test_metrics["mAP"]),
                format_percentage(test_metrics["f1"]),
                format_percentage(test_metrics["accuracy"]),
                format_percentage(test_metrics["jaccard"]),
            )
        )
        epoch_logs.append(
            {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["roc_auc"],
                "train_mAP": train_metrics["mAP"],
                "train_f1": train_metrics["f1"],
                "train_accuracy": train_metrics["accuracy"],
                "train_jaccard": train_metrics["jaccard"],
                "test_loss": test_metrics["loss"],
                "test_auc": test_metrics["roc_auc"],
                "test_mAP": test_metrics["mAP"],
                "test_f1": test_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_jaccard": test_metrics["jaccard"],
                "test_window_count": test_window_metrics["num_windows"],
                "test_window_auc": test_window_metrics["roc_auc"],
                "test_window_mAP": test_window_metrics["mAP"],
                "test_window_f1": test_window_metrics["f1"],
                "test_window_accuracy": test_window_metrics["accuracy"],
                "test_window_jaccard": test_window_metrics["jaccard"],
            }
        )

        current_auc_key = metric_key(test_metrics["roc_auc"])
        current_map_key = metric_key(test_metrics["mAP"])
        best_auc_key = metric_key(best_test_metrics["roc_auc"]) if best_test_metrics is not None else float("-inf")
        best_map_key = metric_key(best_test_metrics["mAP"]) if best_test_metrics is not None else float("-inf")

        if current_auc_key > best_auc_key or (current_auc_key == best_auc_key and current_map_key > best_map_key):
            best_epoch = epoch
            best_test_metrics = dict(test_metrics)
            best_test_window_metrics = dict(test_window_metrics)
            best_model_wts = copy.deepcopy(model.state_dict())
            best_outputs = {
                "preds": list(test_metrics["all_preds"]),
                "scores": list(test_metrics["all_scores"]),
                "labels": list(test_metrics["all_labels"]),
                "video_names": list(test_metrics["video_names"]),
                "video_lengths": list(test_metrics["video_lengths"]),
            }
            torch.save(best_model_wts, os.path.join(fold_dir, "model_best.pth"))
            print(
                "updated best model: fold={} epoch={} AUC={} mAP={}".format(
                    fold_id,
                    best_epoch,
                    format_percentage(best_test_metrics["roc_auc"]),
                    format_percentage(best_test_metrics["mAP"]),
                )
            )

    with open(os.path.join(fold_dir, "training_log.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_logs[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_logs)

    if best_outputs is not None:
        save_video_outputs(
            fold_dir,
            best_outputs["video_names"],
            best_outputs["video_lengths"],
            best_outputs["preds"],
            best_outputs["scores"],
            best_outputs["labels"],
        )

    fold_summary = {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "num_train_videos": len(train_dataset),
        "num_test_videos": len(test_dataset),
        "tasks": task_stats,
        "test_window_config": {
            "window_length": args.test_window_length,
            "window_stride": args.test_window_stride,
            "score_aggregation": "max",
            "label_aggregation": "any_error",
        },
        "test_metrics": serialize_metrics(best_test_metrics or {}),
        "test_window_metrics": serialize_metrics(best_test_window_metrics or {}),
    }

    with open(os.path.join(fold_dir, "metrics_summary.json"), "w") as handle:
        json.dump(fold_summary, handle, indent=2)

    print(
        "best fold={} epoch={} AUC={} mAP={} F1={} Acc={} Jaccard={}".format(
            fold_id,
            best_epoch,
            format_percentage(fold_summary["test_metrics"].get("roc_auc", float("nan"))),
            format_percentage(fold_summary["test_metrics"].get("mAP", float("nan"))),
            format_percentage(fold_summary["test_metrics"].get("f1", float("nan"))),
            format_percentage(fold_summary["test_metrics"].get("accuracy", float("nan"))),
            format_percentage(fold_summary["test_metrics"].get("jaccard", float("nan"))),
        )
    )
    print(
        "best fold={} epoch={} window AUC={} mAP={} F1={} Acc={} Jaccard={} n={}".format(
            fold_id,
            best_epoch,
            format_percentage(fold_summary["test_window_metrics"].get("roc_auc", float("nan"))),
            format_percentage(fold_summary["test_window_metrics"].get("mAP", float("nan"))),
            format_percentage(fold_summary["test_window_metrics"].get("f1", float("nan"))),
            format_percentage(fold_summary["test_window_metrics"].get("accuracy", float("nan"))),
            format_percentage(fold_summary["test_window_metrics"].get("jaccard", float("nan"))),
            fold_summary["test_window_metrics"].get("num_windows", 0),
        )
    )
    return fold_summary


def aggregate_fold_metrics(fold_summaries: Sequence[Dict[str, object]]) -> Dict[str, float]:
    metric_names = ["roc_auc", "mAP", "f1", "accuracy", "jaccard"]
    aggregate = {}
    metric_groups = {
        "": "test_metrics",
        "window_": "test_window_metrics",
    }
    for prefix, summary_key in metric_groups.items():
        for metric_name in metric_names:
            values = [
                float(summary.get(summary_key, {}).get(metric_name))
                for summary in fold_summaries
                if metric_name in summary.get(summary_key, {})
                and np.isfinite(summary.get(summary_key, {}).get(metric_name))
            ]
            if values:
                aggregate[prefix + metric_name + "_mean"] = float(np.mean(values))
                aggregate[prefix + metric_name + "_std"] = float(np.std(values))
            else:
                aggregate[prefix + metric_name + "_mean"] = float("nan")
                aggregate[prefix + metric_name + "_std"] = float("nan")
    return aggregate


def main(args):
    set_seed(args.seed)
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    fold_ids = tuple(args.folds) if args.folds else get_split_fold_ids(args.split_scheme)

    first_train_records, _, _ = collect_fold_records(args.tasks, args.split_scheme, fold_ids[0], args.split_root)
    feature_dim = args.features_dim or infer_feature_dim(args.pkl_root, first_train_records)
    args.run_dir = os.path.join(args.output_root, str(args.lr), args.exp)
    os.makedirs(args.run_dir, exist_ok=True)

    print("experiment name   :", args.exp)
    print("split scheme      :", args.split_scheme)
    print("folds             :", ",".join(str(fold_id) for fold_id in fold_ids))
    print("tasks             :", ",".join(args.tasks))
    print("pkl root          :", args.pkl_root)
    print("split root        :", args.split_root)
    print("feature dim       :", feature_dim)
    print("epochs            :", args.epoch)
    print("workers           :", args.work)
    print("learning rate     :", args.lr)
    print("device            :", device)
    print("seed              :", args.seed)
    print("test window len   :", args.test_window_length)
    print("test window stride:", args.test_window_stride)

    fold_summaries = []
    for fold_id in fold_ids:
        fold_summaries.append(train_fold(args, fold_id, device, feature_dim))

    aggregate = aggregate_fold_metrics(fold_summaries)
    crossval_summary = {
        "split_scheme": args.split_scheme,
        "tasks": list(args.tasks),
        "test_window_config": {
            "window_length": args.test_window_length,
            "window_stride": args.test_window_stride,
            "score_aggregation": "max",
            "label_aggregation": "any_error",
        },
        "folds": fold_summaries,
        "aggregate": aggregate,
    }

    with open(os.path.join(args.run_dir, "crossval_summary.json"), "w") as handle:
        json.dump(crossval_summary, handle, indent=2)

    with open(os.path.join(args.run_dir, "crossval_metrics.csv"), "w", newline="") as handle:
        fieldnames = [
            "fold",
            "best_epoch",
            "roc_auc",
            "mAP",
            "f1",
            "accuracy",
            "jaccard",
            "window_num_windows",
            "window_roc_auc",
            "window_mAP",
            "window_f1",
            "window_accuracy",
            "window_jaccard",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in fold_summaries:
            writer.writerow(
                {
                    "fold": summary["fold"],
                    "best_epoch": summary["best_epoch"],
                    "roc_auc": summary["test_metrics"].get("roc_auc"),
                    "mAP": summary["test_metrics"].get("mAP"),
                    "f1": summary["test_metrics"].get("f1"),
                    "accuracy": summary["test_metrics"].get("accuracy"),
                    "jaccard": summary["test_metrics"].get("jaccard"),
                    "window_num_windows": summary["test_window_metrics"].get("num_windows"),
                    "window_roc_auc": summary["test_window_metrics"].get("roc_auc"),
                    "window_mAP": summary["test_window_metrics"].get("mAP"),
                    "window_f1": summary["test_window_metrics"].get("f1"),
                    "window_accuracy": summary["test_window_metrics"].get("accuracy"),
                    "window_jaccard": summary["test_window_metrics"].get("jaccard"),
                }
            )

    print("\n=== Cross-validation summary ===")
    for metric_name in ["roc_auc", "mAP", "f1", "accuracy", "jaccard"]:
        print(
            "{} mean/std: {} / {}".format(
                metric_name,
                format_percentage(aggregate[metric_name + "_mean"]),
                format_percentage(aggregate[metric_name + "_std"]),
            )
        )
    print(
        "window config: len={} stride={} score=max label=any_error".format(
            args.test_window_length,
            args.test_window_stride,
        )
    )
    for metric_name in ["roc_auc", "mAP", "f1", "accuracy", "jaccard"]:
        print(
            "window_{} mean/std: {} / {}".format(
                metric_name,
                format_percentage(aggregate["window_" + metric_name + "_mean"]),
                format_percentage(aggregate["window_" + metric_name + "_std"]),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEDMamba LOSO training on transformed JIGSAWS PKLs")
    parser.add_argument("-exp", default="SEDMamba-JIGSAWS", type=str, help="experiment name")
    parser.add_argument("--pkl_root", default=DEFAULT_PKL_ROOT, type=str, help="root directory of transformed JIGSAWS PKLs")
    parser.add_argument("--split_root", default=DEFAULT_SPLIT_ROOT, type=str, help="root directory containing LOSO split CSVs")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, type=str, help="root output directory for logs and checkpoints")
    parser.add_argument("--split_scheme", default="loso", choices=["loso"], type=str, help="cross-validation scheme")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=list(DEFAULT_TASKS), help="tasks to include")
    parser.add_argument("--folds", nargs="*", default=None, type=int, help="optional subset of LOSO folds to run")
    parser.add_argument("-gpu_id", type=str, nargs="?", default="cuda:0", help="device id to run")
    parser.add_argument("-w", "--work", default=4, type=int, help="num of workers to use")
    parser.add_argument("-s", "--seed", default=2, type=int, help="random seed")
    parser.add_argument("-e", "--epoch", default=200, type=int, help="epochs to train and evaluate")
    parser.add_argument("-l", "--lr", default=1e-4, type=float, help="learning rate for optimizer")
    parser.add_argument("-cls", "--num_class", default=1, type=int, help="num of classes")
    parser.add_argument("-fd", "--features_dim", default=0, type=int, help="feature dimension; 0 means infer from PKLs")
    parser.add_argument("-nb", "--num_block", default=3, type=int, help="num of BMSS blocks")
    parser.add_argument("-g", "--com_factor", default=64, type=int, help="compression factor G")
    parser.add_argument("--test_window_length", default=DEFAULT_TEST_WINDOW_LENGTH, type=int, help="window length for additional test-set window metrics")
    parser.add_argument("--test_window_stride", default=DEFAULT_TEST_WINDOW_STRIDE, type=int, help="window stride for additional test-set window metrics")

    parsed_args = parser.parse_args()
    logger = CompleteLogger(os.path.join(parsed_args.output_root, str(parsed_args.lr), parsed_args.exp))
    try:
        main(parsed_args)
        print("Done")
    finally:
        logger.close()
