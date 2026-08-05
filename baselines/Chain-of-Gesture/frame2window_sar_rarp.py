import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score


def _read_prediction_rows(csv_path: str) -> List[Dict[str, float]]:
    rows = []
    with open(csv_path, "r") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            score = float(row[3]) if len(row) > 3 else float(row[2])
            rows.append(
                {
                    "gesture": int(row[0]),
                    "true": int(row[1]),
                    "pred": int(row[2]),
                    "score": score,
                }
            )
    return rows


def _split_gesture_segments(rows: Sequence[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    if not rows:
        return []
    segments = []
    current_segment = [rows[0]]

    for row in rows[1:]:
        if row["gesture"] != current_segment[-1]["gesture"]:
            segments.append(current_segment)
            current_segment = [row]
        else:
            current_segment.append(row)

    if current_segment:
        segments.append(current_segment)
    return segments


def _window_slices(segment_length: int, window_size: int, stride: int) -> List[slice]:
    if segment_length <= window_size:
        return [slice(0, segment_length)]

    start = (segment_length - window_size) % stride
    return [
        slice(window_start, window_start + window_size)
        for window_start in range(start, segment_length - window_size + 1, stride)
    ]


def _compute_metrics(window_df: pd.DataFrame) -> Dict[str, float]:
    if window_df.empty:
        return {
            "f1": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "jaccard": 0.0,
            "roc_auc": float("nan"),
        }

    y_true = window_df["true"].astype(int)
    y_pred = window_df["pred"].astype(int)
    y_score = window_df["score"].astype(float)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "jaccard": float(jaccard_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "roc_auc": float("nan"),
    }
    if y_true.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate window-level metrics from SAR_RARP50 COG predictions")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory produced by train_COG_sar_rarp.py")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold. Defaults to best_threshold from metrics_summary.json when available.",
    )
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None:
        metrics_summary_path = os.path.join(args.pred_dir, "metrics_summary.json")
        if os.path.isfile(metrics_summary_path):
            with open(metrics_summary_path, "r") as handle:
                metrics_summary = json.load(handle)
            threshold = float(metrics_summary.get("best_threshold", 0.5))
        else:
            threshold = 0.5

    video_csv_paths = sorted(glob.glob(os.path.join(args.pred_dir, "video_*.csv")))
    if not video_csv_paths:
        raise FileNotFoundError("No per-video prediction CSVs found in {}".format(args.pred_dir))

    window_rows = []
    for video_csv_path in video_csv_paths:
        video_id = os.path.splitext(os.path.basename(video_csv_path))[0]
        rows = _read_prediction_rows(video_csv_path)
        segments = _split_gesture_segments(rows)
        for segment_index, segment_rows in enumerate(segments):
            gesture_id = int(segment_rows[0]["gesture"])
            for window_index, window_slice in enumerate(
                _window_slices(len(segment_rows), args.window_size, args.stride)
            ):
                current_rows = segment_rows[window_slice]
                scores = [row["score"] for row in current_rows]
                labels = [row["true"] for row in current_rows]
                start_frame = window_slice.start
                end_frame = window_slice.stop - 1
                mean_score = float(np.mean(scores))
                pred_label = 1 if mean_score > threshold else 0
                true_label = int(max(labels))
                window_rows.append(
                    {
                        "video_id": video_id,
                        "segment_index": segment_index,
                        "window_index": window_index,
                        "gesture": gesture_id,
                        "true": true_label,
                        "pred": pred_label,
                        "score": mean_score,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )

    window_df = pd.DataFrame(window_rows)
    output_path = os.path.join(args.pred_dir, "window_true_pred.csv")
    window_df.to_csv(output_path, index=False)

    metrics = _compute_metrics(window_df)
    metrics["num_windows"] = int(len(window_df))
    metrics["threshold"] = float(threshold)
    print("pred_dir        :", args.pred_dir)
    print("threshold       :", "{:.6f}".format(threshold))
    print("num_windows     :", metrics["num_windows"])
    print("window_f1       :", "{:.6f}".format(metrics["f1"]))
    print("window_accuracy :", "{:.6f}".format(metrics["accuracy"]))
    print("window_precision:", "{:.6f}".format(metrics["precision"]))
    print("window_recall   :", "{:.6f}".format(metrics["recall"]))
    print("window_jaccard  :", "{:.6f}".format(metrics["jaccard"]))
    print(
        "window_roc_auc  :",
        "{:.6f}".format(metrics["roc_auc"]) if not np.isnan(metrics["roc_auc"]) else "nan",
    )

    metrics_path = os.path.join(args.pred_dir, "window_metrics.json")
    with open(metrics_path, "w") as handle:
        import json

        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
