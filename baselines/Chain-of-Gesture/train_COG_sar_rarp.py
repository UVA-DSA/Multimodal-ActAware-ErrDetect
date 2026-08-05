import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import optim
from torch.utils.data import DataLoader

import models
from dataload_sar_rarp import SarRarpVideoDataset, list_video_ids
from sar_rarp_prompts import SAR_RARP_GESTURE_PROMPTS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "SAR_RARP50"))
DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "exp_log_sar_rarp")
DEFAULT_GESTURE_PROMPT_PATH = os.path.join(SCRIPT_DIR, "utils", "gest_prompt_sar_rarp_B32.pt")
NUM_REFINE_STAGES = 3
TEST_WINDOW_LENGTH = 10
TEST_WINDOW_STRIDE = 6


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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def split_train_val(video_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    video_ids = list(video_ids)
    if not video_ids:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    val_count = max(1, int(round(len(video_ids) * val_ratio))) if len(video_ids) > 1 else 0
    val_ids = sorted(video_ids[:val_count])
    train_ids = sorted(video_ids[val_count:])
    return train_ids, val_ids


def infer_feature_dim(dataset: SarRarpVideoDataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    sample_features, _, _, _, _ = dataset[0]
    return int(sample_features.shape[-1])


def infer_min_sequence_length(dataset: SarRarpVideoDataset) -> int:
    if len(dataset) == 0:
        return 0

    min_length = None
    for idx in range(len(dataset)):
        _, num_frames, _, _, _ = dataset[idx]
        num_frames = int(num_frames)
        if min_length is None or num_frames < min_length:
            min_length = num_frames
    return int(min_length or 0)


def _pooled_length(length: int, kernel_size: int) -> int:
    if kernel_size < 1:
        raise ValueError("Pooling kernel size must be positive, got {}.".format(kernel_size))
    if length < kernel_size:
        return 0
    return 1 + (length - kernel_size) // kernel_size


def max_safe_refine_kernel(min_length: int, num_refine_stages: int) -> int:
    if min_length < 1:
        return 0

    max_kernel = 0
    for kernel_size in range(1, min_length + 1):
        pooled_length = min_length
        for _ in range(num_refine_stages):
            pooled_length = _pooled_length(pooled_length, kernel_size)
            if pooled_length < 1:
                break
        else:
            max_kernel = kernel_size
    return max_kernel


def validate_refinement_pool_kernel(
    pool_kernel: int,
    datasets: Sequence[SarRarpVideoDataset],
    num_refine_stages: int,
) -> Tuple[int, int]:
    if pool_kernel < 1:
        raise ValueError("--train must be a positive integer. Got {}.".format(pool_kernel))

    min_lengths = [infer_min_sequence_length(dataset) for dataset in datasets if len(dataset) > 0]
    if not min_lengths:
        raise ValueError("Cannot validate refinement pooling without at least one non-empty dataset split.")

    shortest_clip_length = min(min_lengths)
    max_kernel = max_safe_refine_kernel(shortest_clip_length, num_refine_stages)
    if pool_kernel > max_kernel:
        raise ValueError(
            "Invalid --train value: {}. In this script, --train is the hierarchical "
            "refinement pooling kernel, not the batch size. With {} refinement stages "
            "and the shortest clip length of {}, the largest safe value is {}. "
            "Use --train 1 to match the original COG default.".format(
                pool_kernel,
                num_refine_stages,
                shortest_clip_length,
                max_kernel,
            )
        )
    return shortest_clip_length, max_kernel


def build_model(args, feature_dim: int, device: torch.device) -> nn.Module:
    out_features = 2
    num_layers_basic = 11
    num_refine_layers = args.layers
    num_refine_stages = NUM_REFINE_STAGES
    mstcn_f_maps = 64
    d_model = args.dmodel
    d_q = int(d_model / 8)

    model = models.COG(
        args,
        num_layers_basic,
        num_refine_layers,
        num_refine_stages,
        mstcn_f_maps,
        feature_dim,
        out_features,
        True,
        d_model,
        d_q,
        args.len_q,
        device,
        gest_prompt=args.gesture_prompt_path,
        gesture_prompts=SAR_RARP_GESTURE_PROMPTS,
    )
    model.to(device)
    return model


def compute_stage_losses(
    predicted_list: List[torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    criterion2: nn.Module,
    smooth_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, stage_outputs, labels_list = models.fusion(predicted_list, labels)
    classification_loss = torch.tensor(0.0, device=labels.device)
    smooth_loss = torch.tensor(0.0, device=labels.device)

    for stage_output, stage_labels in zip(stage_outputs, labels_list):
        stage_classes = stage_output.squeeze(0).transpose(1, 0)
        ce_loss = criterion(stage_classes.squeeze(), stage_labels)
        classification_loss = classification_loss + ce_loss

        if stage_classes.size(0) > 1:
            temporal_loss = criterion2(
                F.log_softmax(stage_classes[1:, :], dim=1),
                F.log_softmax(stage_classes.detach()[:-1, :], dim=1),
            )
            temporal_loss = torch.mean(torch.clamp(temporal_loss, min=0, max=16))
            smooth_loss = smooth_loss + temporal_loss

    num_stages = max(len(stage_outputs), 1)
    classification_loss = classification_loss / float(num_stages)
    smooth_loss = smooth_loss / float(num_stages)
    total_loss = classification_loss + smooth_weight * smooth_loss
    return total_loss, classification_loss, smooth_loss


def _safe_binary_metrics(labels: List[int], scores: List[float], preds: List[int]) -> Dict[str, object]:
    labels_np = np.asarray(labels).astype(int)
    preds_np = np.asarray(preds).astype(int)
    scores_np = np.asarray(scores, dtype=np.float32)

    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds_np)) if labels_np.size else 0.0,
        "f1": float(f1_score(labels_np, preds_np, average="binary", pos_label=1, zero_division=0)) if labels_np.size else 0.0,
        "jaccard": float(jaccard_score(labels_np, preds_np, average="binary", pos_label=1, zero_division=0)) if labels_np.size else 0.0,
        "precision": float(precision_score(labels_np, preds_np, average="binary", pos_label=1, zero_division=0)) if labels_np.size else 0.0,
        "recall": float(recall_score(labels_np, preds_np, average="binary", pos_label=1, zero_division=0)) if labels_np.size else 0.0,
        "precision_each": precision_score(labels_np, preds_np, average=None, zero_division=0).tolist() if labels_np.size else [],
        "recall_each": recall_score(labels_np, preds_np, average=None, zero_division=0).tolist() if labels_np.size else [],
        "class_report": classification_report(labels_np, preds_np, labels=[0, 1], digits=6, zero_division=0) if labels_np.size else "",
        "confusion_matrix": confusion_matrix(labels_np, preds_np, labels=[0, 1]).tolist() if labels_np.size else [[0, 0], [0, 0]],
        "roc_auc": float("nan"),
        "fpr": [],
        "tpr": [],
    }

    if labels_np.size and np.unique(labels_np).size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels_np, scores_np))
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        metrics["fpr"] = fpr.tolist()
        metrics["tpr"] = tpr.tolist()

    return metrics


def _find_best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> Tuple[float, Dict[str, object]]:
    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(scores, dtype=np.float32)
    if labels_np.size == 0:
        empty_metrics = _safe_binary_metrics([], [], [])
        empty_metrics["threshold"] = 0.5
        return 0.5, empty_metrics

    best_threshold = 0.5
    best_f1 = -1.0
    total_pos = int(labels_np.sum())
    if total_pos == 0:
        best_metrics = _apply_threshold_metrics(labels_np.tolist(), scores_np.tolist(), 1.0)
        return 1.0, best_metrics

    # Sweep unique score groups once instead of recomputing full reports for
    # every threshold candidate. This keeps per-epoch threshold tuning cheap.
    order = np.argsort(scores_np)[::-1]
    sorted_scores = scores_np[order]
    sorted_labels = labels_np[order]

    tp = 0
    fp = 0
    fn = total_pos
    index = 0
    num_scores = len(sorted_scores)

    while index < num_scores:
        current_score = float(sorted_scores[index])
        next_index = index
        while next_index < num_scores and float(sorted_scores[next_index]) == current_score:
            if int(sorted_labels[next_index]) == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
            next_index += 1

        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        current_f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if next_index < num_scores:
            threshold = float(sorted_scores[next_index])
        else:
            threshold = float(np.nextafter(current_score, -np.inf))

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

        index = next_index

    best_metrics = _apply_threshold_metrics(labels_np.tolist(), scores_np.tolist(), best_threshold)
    return best_threshold, best_metrics


def _apply_threshold_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, object]:
    scores_np = np.asarray(scores, dtype=np.float32)
    preds = (scores_np > threshold).astype(int)
    metrics = _safe_binary_metrics(list(labels), scores_np.tolist(), preds.tolist())
    metrics["threshold"] = float(threshold)
    return metrics


def _positive_rate(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    values_np = np.asarray(values).astype(int)
    return float(values_np.mean())


def _score_mean_for_class(labels: Sequence[int], scores: Sequence[float], class_value: int) -> float:
    if not labels:
        return float("nan")
    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(scores, dtype=np.float32)
    class_scores = scores_np[labels_np == class_value]
    if class_scores.size == 0:
        return float("nan")
    return float(class_scores.mean())


def _metric_snapshot(metrics: Dict[str, object]) -> Dict[str, object]:
    keep_keys = [
        "loss",
        "accuracy",
        "f1",
        "jaccard",
        "precision",
        "recall",
        "roc_auc",
        "confusion_matrix",
        "class_report",
        "threshold",
        "elapsed_time_sec",
        "num_samples",
        "num_windows",
        "window_length",
        "window_stride",
        "score_aggregation",
        "label_aggregation",
    ]
    snapshot = {}
    for key in keep_keys:
        if key in metrics:
            snapshot[key] = metrics[key]
    return snapshot


def _retag_predictions_with_threshold(
    video_predictions: Sequence[Dict[str, object]],
    threshold: float,
) -> List[Dict[str, object]]:
    adjusted_predictions = []
    for item in video_predictions:
        scores = [float(score) for score in item["scores"]]
        preds = [1 if score > threshold else 0 for score in scores]
        adjusted_predictions.append(
            {
                "video_id": item["video_id"],
                "gestures": list(item["gestures"]),
                "labels": list(item["labels"]),
                "preds": preds,
                "scores": scores,
            }
        )
    return adjusted_predictions


def _flatten_recorded_predictions(video_predictions: Sequence[Dict[str, object]]) -> Dict[str, List[float]]:
    flattened = {
        "scores": [],
        "preds": [],
        "labels": [],
        "gestures": [],
        "video_lengths": [],
    }

    for item in video_predictions:
        scores = [float(score) for score in item["scores"]]
        preds = [int(pred) for pred in item["preds"]]
        labels = [int(label) for label in item["labels"]]
        gestures = [int(gesture) for gesture in item.get("gestures", [])]

        if not (len(scores) == len(preds) == len(labels)):
            raise ValueError(
                "Mismatched recorded prediction lengths for {}.".format(item.get("video_id", "<unknown>"))
            )

        flattened["scores"].extend(scores)
        flattened["preds"].extend(preds)
        flattened["labels"].extend(labels)
        flattened["gestures"].extend(gestures)
        flattened["video_lengths"].append(len(labels))

    return flattened


def _compute_window_samples(
    video_predictions: Sequence[Dict[str, object]],
    window_length: int = TEST_WINDOW_LENGTH,
    window_stride: int = TEST_WINDOW_STRIDE,
) -> Tuple[List[int], List[float]]:
    if window_length < 1:
        raise ValueError("window_length must be positive, got {}.".format(window_length))
    if window_stride < 1:
        raise ValueError("window_stride must be positive, got {}.".format(window_stride))

    window_labels: List[int] = []
    window_scores: List[float] = []

    for item in video_predictions:
        video_scores = np.asarray(item["scores"], dtype=np.float32)
        video_labels = np.asarray(item["labels"]).astype(int)
        if video_scores.size != video_labels.size:
            raise ValueError(
                "Mismatched score/label lengths for {}.".format(item.get("video_id", "<unknown>"))
            )

        if video_labels.size < window_length:
            continue

        for window_start in range(0, video_labels.size - window_length + 1, window_stride):
            window_end = window_start + window_length
            cur_scores = video_scores[window_start:window_end]
            cur_labels = video_labels[window_start:window_end]
            window_scores.append(float(np.max(cur_scores)))
            window_labels.append(int(np.any(cur_labels)))

    return window_labels, window_scores


def _apply_window_threshold_metrics(
    video_predictions: Sequence[Dict[str, object]],
    threshold: float,
    window_length: int = TEST_WINDOW_LENGTH,
    window_stride: int = TEST_WINDOW_STRIDE,
) -> Dict[str, object]:
    window_labels, window_scores = _compute_window_samples(
        video_predictions,
        window_length=window_length,
        window_stride=window_stride,
    )
    metrics = _apply_threshold_metrics(window_labels, window_scores, threshold)
    metrics["num_windows"] = len(window_labels)
    metrics["window_length"] = int(window_length)
    metrics["window_stride"] = int(window_stride)
    metrics["score_aggregation"] = "max"
    metrics["label_aggregation"] = "any_error"
    return metrics


def _evaluate_recorded_test_predictions(
    video_predictions: Sequence[Dict[str, object]],
    tuned_threshold: float,
    loss: float,
    elapsed_time_sec: float,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object], Dict[str, object]]:
    flattened = _flatten_recorded_predictions(video_predictions)
    default_metrics = _apply_threshold_metrics(flattened["labels"], flattened["scores"], 0.5)
    tuned_metrics = _apply_threshold_metrics(flattened["labels"], flattened["scores"], tuned_threshold)

    for metrics in (default_metrics, tuned_metrics):
        metrics["loss"] = float(loss)
        metrics["elapsed_time_sec"] = float(elapsed_time_sec)
        metrics["num_samples"] = len(flattened["labels"])

    window_default_metrics = _apply_window_threshold_metrics(
        video_predictions,
        threshold=0.5,
        window_length=TEST_WINDOW_LENGTH,
        window_stride=TEST_WINDOW_STRIDE,
    )
    window_tuned_metrics = _apply_window_threshold_metrics(
        video_predictions,
        threshold=tuned_threshold,
        window_length=TEST_WINDOW_LENGTH,
        window_stride=TEST_WINDOW_STRIDE,
    )

    for metrics in (window_default_metrics, window_tuned_metrics):
        metrics["elapsed_time_sec"] = float(elapsed_time_sec)

    return default_metrics, tuned_metrics, window_default_metrics, window_tuned_metrics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    criterion2: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    smooth_weight: float,
    train_mode: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if train_mode:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    all_scores = []
    all_preds = []
    all_labels = []
    all_gestures = []
    video_predictions = []
    start_time = time.time()

    for batch_idx, data in enumerate(dataloader):
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        video_features = data[0].to(device)
        error_labels = data[2].squeeze(0).to(device)
        gesture_labels = data[3].squeeze(0)
        video_id = data[4][0]

        with torch.set_grad_enabled(train_mode):
            predicted_list, _ = model(video_features)
            total_loss, _, _ = compute_stage_losses(
                predicted_list,
                error_labels,
                criterion,
                criterion2,
                smooth_weight,
            )
            if train_mode:
                total_loss.backward()
                optimizer.step()

        resized_logits = F.interpolate(
            predicted_list[0],
            size=error_labels.size(0),
            mode="nearest",
        ).squeeze(0).transpose(1, 0)
        probabilities = torch.softmax(resized_logits, dim=1)[:, 1].detach().cpu().numpy()
        predictions = torch.argmax(resized_logits, dim=1).detach().cpu().numpy().astype(int)
        labels_np = error_labels.detach().cpu().numpy().astype(int)
        gestures_np = gesture_labels.detach().cpu().numpy().astype(int)

        all_scores.extend(probabilities.tolist())
        all_preds.extend(predictions.tolist())
        all_labels.extend(labels_np.tolist())
        all_gestures.extend(gestures_np.tolist())
        epoch_loss += float(total_loss.item())

        video_predictions.append(
            {
                "video_id": video_id,
                "gestures": gestures_np.tolist(),
                "labels": labels_np.tolist(),
                "preds": predictions.tolist(),
                "scores": probabilities.tolist(),
            }
        )

        progress = batch_idx + 1
        print(
            "{} progress: {} [{}/{}]".format(
                "train" if train_mode else "eval",
                str(round(progress / len(dataloader) * 100, 2)) + "%",
                progress,
                len(dataloader),
            ),
            end="\n" if progress == len(dataloader) else "\r",
        )

    elapsed_time = time.time() - start_time
    metrics = _safe_binary_metrics(all_labels, all_scores, all_preds)
    metrics["loss"] = epoch_loss / max(len(dataloader), 1)
    metrics["elapsed_time_sec"] = elapsed_time
    metrics["all_scores"] = all_scores
    metrics["all_preds"] = all_preds
    metrics["all_labels"] = all_labels
    metrics["all_gestures"] = all_gestures
    return metrics, video_predictions


def save_prediction_files(output_dir: str, video_predictions: Sequence[Dict[str, object]]) -> None:
    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for item in video_predictions:
            rows = zip(item["gestures"], item["labels"], item["preds"], item["scores"])
            for gesture, label, pred, score in rows:
                writer.writerow([gesture, label, pred, score])

    for item in video_predictions:
        video_path = os.path.join(output_dir, item["video_id"] + ".csv")
        with open(video_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            rows = zip(item["gestures"], item["labels"], item["preds"], item["scores"])
            for gesture, label, pred, score in rows:
                writer.writerow([gesture, label, pred, score])


def save_json(path: str, content: Dict[str, object]) -> None:
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train COG on SAR_RARP50")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--feature_source", type=str, choices=["pkl", "npy"], default="npy")
    parser.add_argument("--gpu_id", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_", type=float, default=0.15)
    parser.add_argument("--dmodel", type=int, default=64)
    parser.add_argument("--len_q", type=int, default=40)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="Legacy COG argument used as the hierarchical refinement pooling kernel size (not batch size).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--use_test_as_val", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gesture_prompt_path", type=str, default=DEFAULT_GESTURE_PROMPT_PATH)
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["val_f1", "val_tuned_f1", "val_auc"],
        default="val_tuned_f1",
        help="Metric used to select the best checkpoint.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.gesture_prompt_path), exist_ok=True)
    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")
    use_test_as_val = bool(args.use_test_as_val == 1)

    all_train_ids = list_video_ids(args.data_root, train=True)
    if use_test_as_val:
        train_ids = all_train_ids
        val_ids = []
    else:
        train_ids, val_ids = split_train_val(all_train_ids, args.val_ratio, args.seed)

    test_ids = list_video_ids(args.data_root, train=False)
    train_dataset = SarRarpVideoDataset(
        data_root=args.data_root,
        feature_source=args.feature_source,
        train=True,
        video_ids=train_ids,
    )
    val_dataset = SarRarpVideoDataset(
        data_root=args.data_root,
        feature_source=args.feature_source,
        train=not use_test_as_val,
        video_ids=test_ids if use_test_as_val else val_ids,
    )
    test_dataset = SarRarpVideoDataset(
        data_root=args.data_root,
        feature_source=args.feature_source,
        train=False,
        video_ids=test_ids,
    )

    shortest_clip_length, max_safe_refine_kernel_value = validate_refinement_pool_kernel(
        args.train,
        [train_dataset, val_dataset, test_dataset],
        NUM_REFINE_STAGES,
    )
    feature_dim = infer_feature_dim(train_dataset)
    model = build_model(args, feature_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
    )

    run_name = args.run_name or "cog_sar_rarp_{}_lr{}_seed{}".format(
        args.feature_source,
        args.lr,
        args.seed,
    )
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    split_summary = {
        "train_ids": train_ids,
        "val_ids": test_ids if use_test_as_val else val_ids,
        "test_ids": test_ids,
        "feature_source": args.feature_source,
        "feature_dim": feature_dim,
        "device": str(device),
        "use_test_as_val": use_test_as_val,
    }
    save_json(os.path.join(run_dir, "split_summary.json"), split_summary)

    print("run_name          :", run_name)
    print("device            :", device)
    print("feature_source    :", args.feature_source)
    print("feature_dim       :", feature_dim)
    print("refine_pool_kernel:", args.train)
    print("shortest_clip_len :", shortest_clip_length)
    print("max_safe_kernel   :", max_safe_refine_kernel_value)
    print("num_train_videos  :", len(train_dataset))
    print("num_val_videos    :", len(val_dataset))
    print("num_test_videos   :", len(test_dataset))
    print(
        "test window eval  : len={} stride={} score=max label=any_error (deferred until final best-checkpoint pass)".format(
            TEST_WINDOW_LENGTH,
            TEST_WINDOW_STRIDE,
        )
    )

    best_epoch = -1
    best_selection_score = -1.0
    best_state = None
    best_train_metrics = None
    best_val_metrics = None
    best_test_metrics = None
    best_val_tuned_metrics = None
    best_test_tuned_metrics = None
    best_test_window_metrics = None
    best_test_window_tuned_metrics = None
    best_test_predictions = None
    best_threshold = 0.5
    epochs_without_improvement = 0
    epoch_logs = []

    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            criterion2,
            optimizer,
            device,
            args.lambda_,
            train_mode=True,
        )
        val_metrics, _ = run_epoch(
            model,
            val_loader,
            criterion,
            criterion2,
            optimizer,
            device,
            args.lambda_,
            train_mode=False,
        )

        print(
            "train loss: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f}".format(
                train_metrics["loss"],
                "{:.4f}".format(train_metrics["roc_auc"]) if not np.isnan(train_metrics["roc_auc"]) else "nan",
                train_metrics["accuracy"],
                train_metrics["f1"],
                train_metrics["jaccard"],
            )
        )
        print(
            "val   loss: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f}".format(
                val_metrics["loss"],
                "{:.4f}".format(val_metrics["roc_auc"]) if not np.isnan(val_metrics["roc_auc"]) else "nan",
                val_metrics["accuracy"],
                val_metrics["f1"],
                val_metrics["jaccard"],
            )
        )

        val_threshold, val_tuned_metrics = _find_best_f1_threshold(
            val_metrics["all_labels"],
            val_metrics["all_scores"],
        )

        print(
            "val*  th: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f} pred_pos: {:.4f}".format(
                val_threshold,
                "{:.4f}".format(val_tuned_metrics["roc_auc"]) if not np.isnan(val_tuned_metrics["roc_auc"]) else "nan",
                val_tuned_metrics["accuracy"],
                val_tuned_metrics["f1"],
                val_tuned_metrics["jaccard"],
                _positive_rate((np.asarray(val_metrics["all_scores"]) > val_threshold).astype(int).tolist()),
            )
        )
        print(
            "balance label_pos train/val: {:.4f} / {:.4f}".format(
                _positive_rate(train_metrics["all_labels"]),
                _positive_rate(val_metrics["all_labels"]),
            )
        )
        print(
            "score_mean_pos val: {:.4f} | score_mean_neg val: {:.4f}".format(
                _score_mean_for_class(val_metrics["all_labels"], val_metrics["all_scores"], 1),
                _score_mean_for_class(val_metrics["all_labels"], val_metrics["all_scores"], 0),
            )
        )

        selection_score = {
            "val_f1": float(val_metrics["f1"]),
            "val_tuned_f1": float(val_tuned_metrics["f1"]),
            "val_auc": float(val_metrics["roc_auc"]) if not np.isnan(val_metrics["roc_auc"]) else float("-inf"),
        }[args.selection_metric]

        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["roc_auc"],
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "train_jaccard": train_metrics["jaccard"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["roc_auc"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_jaccard": val_metrics["jaccard"],
                "val_tuned_threshold": val_threshold,
                "val_tuned_acc": val_tuned_metrics["accuracy"],
                "val_tuned_f1": val_tuned_metrics["f1"],
                "val_tuned_jaccard": val_tuned_metrics["jaccard"],
                "selection_score": selection_score,
            }
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            best_val_tuned_metrics = val_tuned_metrics
            best_threshold = val_threshold
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "args": vars(args),
                    "feature_dim": feature_dim,
                    "feature_source": args.feature_source,
                    "best_epoch": best_epoch,
                    "best_selection_score": best_selection_score,
                    "selection_metric": args.selection_metric,
                    "best_threshold": best_threshold,
                },
                os.path.join(run_dir, "model_best.pth"),
            )
            print(
                "updated best model at epoch {} with {} {:.4f} and threshold {:.4f}".format(
                    best_epoch,
                    args.selection_metric,
                    best_selection_score,
                    best_threshold,
                )
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("early stopping triggered after {} epochs without improvement".format(args.patience))
                break

    with open(os.path.join(run_dir, "training_log.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_logs[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_logs)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("\nRunning final test evaluation with best checkpoint...")
        test_recording, test_predictions = run_epoch(
            model,
            test_loader,
            criterion,
            criterion2,
            optimizer,
            device,
            args.lambda_,
            train_mode=False,
        )
        (
            best_test_metrics,
            best_test_tuned_metrics,
            best_test_window_metrics,
            best_test_window_tuned_metrics,
        ) = _evaluate_recorded_test_predictions(
            test_predictions,
            tuned_threshold=best_threshold,
            loss=test_recording["loss"],
            elapsed_time_sec=test_recording["elapsed_time_sec"],
        )
        best_test_predictions = _retag_predictions_with_threshold(test_predictions, best_threshold)
        print(
            "final test  loss: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f}".format(
                best_test_metrics["loss"],
                "{:.4f}".format(best_test_metrics["roc_auc"]) if not np.isnan(best_test_metrics["roc_auc"]) else "nan",
                best_test_metrics["accuracy"],
                best_test_metrics["f1"],
                best_test_metrics["jaccard"],
            )
        )
        print(
            "final test* th: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f}".format(
                best_threshold,
                "{:.4f}".format(best_test_tuned_metrics["roc_auc"]) if not np.isnan(best_test_tuned_metrics["roc_auc"]) else "nan",
                best_test_tuned_metrics["accuracy"],
                best_test_tuned_metrics["f1"],
                best_test_tuned_metrics["jaccard"],
            )
        )
        print(
            "final test-window  n: {} th: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f}".format(
                best_test_window_metrics["num_windows"],
                best_test_window_metrics["threshold"],
                "{:.4f}".format(best_test_window_metrics["roc_auc"])
                if not np.isnan(best_test_window_metrics["roc_auc"])
                else "nan",
                best_test_window_metrics["accuracy"],
                best_test_window_metrics["f1"],
                best_test_window_metrics["jaccard"],
            )
        )
        print(
            "final test-window* n: {} th: {:.4f} auc: {} acc: {:.4f} f1: {:.4f} jaccard: {:.4f}".format(
                best_test_window_tuned_metrics["num_windows"],
                best_test_window_tuned_metrics["threshold"],
                "{:.4f}".format(best_test_window_tuned_metrics["roc_auc"])
                if not np.isnan(best_test_window_tuned_metrics["roc_auc"])
                else "nan",
                best_test_window_tuned_metrics["accuracy"],
                best_test_window_tuned_metrics["f1"],
                best_test_window_tuned_metrics["jaccard"],
            )
        )
    save_prediction_files(run_dir, best_test_predictions or [])

    metrics_summary = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_selection_score,
        "best_threshold": best_threshold,
        "test_window_config": {
            "window_length": TEST_WINDOW_LENGTH,
            "window_stride": TEST_WINDOW_STRIDE,
            "score_aggregation": "max",
            "label_aggregation": "any_error",
        },
        "train_metrics": _metric_snapshot(best_train_metrics or {}),
        "val_metrics_default_threshold": _metric_snapshot(best_val_metrics or {}),
        "val_metrics_tuned_threshold": _metric_snapshot(best_val_tuned_metrics or {}),
        "test_metrics_default_threshold": _metric_snapshot(best_test_metrics or {}),
        "test_metrics_tuned_threshold": _metric_snapshot(best_test_tuned_metrics or {}),
        "test_window_metrics_default_threshold": _metric_snapshot(best_test_window_metrics or {}),
        "test_window_metrics_tuned_threshold": _metric_snapshot(best_test_window_tuned_metrics or {}),
    }
    save_json(os.path.join(run_dir, "metrics_summary.json"), metrics_summary)

    print("\nBest epoch:", best_epoch)
    if best_test_tuned_metrics is not None:
        print("Best threshold    :", "{:.6f}".format(best_threshold))
        print("Best test f1      :", "{:.6f}".format(best_test_tuned_metrics["f1"]))
        print("Best test acc     :", "{:.6f}".format(best_test_tuned_metrics["accuracy"]))
        print("Best test jaccard :", "{:.6f}".format(best_test_tuned_metrics["jaccard"]))
        print(
            "Best test auc     :",
            "{:.6f}".format(best_test_tuned_metrics["roc_auc"]) if not np.isnan(best_test_tuned_metrics["roc_auc"]) else "nan",
        )
        print(
            "Best test window f1      :",
            "{:.6f}".format(best_test_window_tuned_metrics["f1"]) if best_test_window_tuned_metrics is not None else "nan",
        )
        print(
            "Best test window acc     :",
            "{:.6f}".format(best_test_window_tuned_metrics["accuracy"]) if best_test_window_tuned_metrics is not None else "nan",
        )
        print(
            "Best test window jaccard :",
            "{:.6f}".format(best_test_window_tuned_metrics["jaccard"]) if best_test_window_tuned_metrics is not None else "nan",
        )
        print(
            "Best test window auc     :",
            "{:.6f}".format(best_test_window_tuned_metrics["roc_auc"])
            if best_test_window_tuned_metrics is not None and not np.isnan(best_test_window_tuned_metrics["roc_auc"])
            else "nan",
        )
        print("Saved outputs to  :", run_dir)


if __name__ == "__main__":
    main()
