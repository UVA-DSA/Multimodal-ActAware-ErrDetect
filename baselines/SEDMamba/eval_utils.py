import csv
import os
import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


def make_worker_init_fn(seed: int):
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn


def metric_key(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def format_percentage(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return "{:.4f}%".format(float(value) * 100.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _binary_confusion_counts(labels: np.ndarray, preds: np.ndarray) -> Dict[str, int]:
    labels_bool = labels.astype(bool)
    preds_bool = preds.astype(bool)
    tp = int(np.sum(labels_bool & preds_bool))
    tn = int(np.sum(~labels_bool & ~preds_bool))
    fp = int(np.sum(~labels_bool & preds_bool))
    fn = int(np.sum(labels_bool & ~preds_bool))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]

    idx = 0
    while idx < len(sorted_scores):
        current_score = sorted_scores[idx]
        group_tp = 0
        group_fp = 0
        while idx < len(sorted_scores) and sorted_scores[idx] == current_score:
            if sorted_labels[idx] == 1:
                group_tp += 1
            else:
                group_fp += 1
            idx += 1
        tp += group_tp
        fp += group_fp
        tpr.append(_safe_ratio(tp, positives))
        fpr.append(_safe_ratio(fp, negatives))

    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) * 0.5
    return float(auc)


def _binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = 0
    fp = 0
    prev_recall = 0.0
    average_precision = 0.0

    idx = 0
    while idx < len(sorted_scores):
        current_score = sorted_scores[idx]
        group_tp = 0
        group_fp = 0
        while idx < len(sorted_scores) and sorted_scores[idx] == current_score:
            if sorted_labels[idx] == 1:
                group_tp += 1
            else:
                group_fp += 1
            idx += 1
        tp += group_tp
        fp += group_fp

        recall = _safe_ratio(tp, positives)
        precision = _safe_ratio(tp, tp + fp)
        average_precision += precision * (recall - prev_recall)
        prev_recall = recall

    return float(average_precision)


def compute_binary_metrics(
    labels: Sequence[float],
    scores: Sequence[float],
    preds: Sequence[float],
) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(scores, dtype=np.float32)
    preds_np = np.asarray(preds).astype(int)

    metrics = {
        "accuracy": 0.0,
        "f1": 0.0,
        "jaccard": 0.0,
        "roc_auc": float("nan"),
        "mAP": float("nan"),
    }

    if labels_np.size == 0:
        return metrics

    counts = _binary_confusion_counts(labels_np, preds_np)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    metrics["accuracy"] = _safe_ratio(tp + tn, tp + tn + fp + fn)
    metrics["f1"] = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    metrics["jaccard"] = _safe_ratio(tp, tp + fp + fn)

    unique_labels = np.unique(labels_np)
    if unique_labels.size >= 2:
        metrics["roc_auc"] = _binary_roc_auc(labels_np, scores_np)
        metrics["mAP"] = _binary_average_precision(labels_np, scores_np)

    return metrics


def compute_window_binary_metrics(
    scores: Sequence[float],
    labels: Sequence[float],
    video_lengths: Sequence[int],
    window_length: int = 10,
    window_stride: int = 6,
) -> Tuple[Dict[str, float], int]:
    if window_length < 1:
        raise ValueError("window_length must be positive, got {}.".format(window_length))
    if window_stride < 1:
        raise ValueError("window_stride must be positive, got {}.".format(window_stride))

    scores_np = np.asarray(scores, dtype=np.float32)
    labels_np = np.asarray(labels).astype(int)
    if scores_np.size != labels_np.size:
        raise ValueError(
            "scores and labels must have the same length, got {} and {}.".format(
                scores_np.size,
                labels_np.size,
            )
        )

    window_scores = []
    window_labels = []
    start_idx = 0

    for video_length in video_lengths:
        video_length = int(video_length)
        end_idx = start_idx + video_length
        if end_idx > scores_np.size:
            raise ValueError(
                "video_lengths exceed flattened prediction length: tried to slice {} > {}.".format(
                    end_idx,
                    scores_np.size,
                )
            )

        video_scores = scores_np[start_idx:end_idx]
        video_labels = labels_np[start_idx:end_idx]

        if video_length >= window_length:
            for window_start in range(0, video_length - window_length + 1, window_stride):
                window_end = window_start + window_length
                cur_scores = video_scores[window_start:window_end]
                cur_labels = video_labels[window_start:window_end]
                # Use the peak frame probability as the window-level score.
                window_scores.append(float(np.max(cur_scores)))
                # Mark a window as positive if it contains any error frame.
                window_labels.append(int(np.any(cur_labels)))

        start_idx = end_idx

    if start_idx != scores_np.size:
        raise ValueError(
            "video_lengths cover {} predictions but {} were provided.".format(
                start_idx,
                scores_np.size,
            )
        )

    if not window_scores:
        return compute_binary_metrics([], [], []), 0

    window_preds = [int(score >= 0.5) for score in window_scores]
    return compute_binary_metrics(window_labels, window_scores, window_preds), len(window_scores)


def save_video_outputs(
    output_dir: str,
    video_names: Sequence[str],
    video_lengths: Sequence[int],
    preds: Sequence[float],
    scores: Sequence[float],
    labels: Sequence[float],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    start_idx = 0
    for video_name, video_length in zip(video_names, video_lengths):
        base_name = os.path.splitext(str(video_name))[0]
        preds_filename = os.path.join(output_dir, base_name + ".csv")
        score_filename = os.path.join(output_dir, base_name + "_score.csv")
        label_filename = os.path.join(output_dir, base_name + "_label.csv")

        with open(preds_filename, "w", newline="") as handle:
            writer = csv.writer(handle)
            for offset in range(video_length):
                writer.writerow([preds[start_idx + offset]])

        with open(score_filename, "w", newline="") as handle:
            writer = csv.writer(handle)
            for offset in range(video_length):
                writer.writerow([scores[start_idx + offset]])

        with open(label_filename, "w", newline="") as handle:
            writer = csv.writer(handle)
            for offset in range(video_length):
                writer.writerow([labels[start_idx + offset]])

        start_idx += int(video_length)
