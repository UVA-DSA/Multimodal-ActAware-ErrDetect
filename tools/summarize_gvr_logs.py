#!/usr/bin/env python3
"""
Summarize GVR runs from epoch-level CSV logs under ./gvr_logs.

Each run CSV is written by train_eval_gvr_error_features.py as:
  training_gvrerr_{model}_{prompt_type}_lr_{lr}_wd_{wd}.csv

This script:
  - reads one or more CSVs
  - selects the best epoch per fold (by max test_f1)
  - reports unweighted mean/std across folds for metrics at that best-test epoch
  - reports a frame-weighted aggregate across folds (default weights = JIGSAWS test-frame
    counts per LOSO fold: 8332, 6056, 7066, 6979, 5433 for folds 1..5)
  - supports LOUO runs whose fold IDs may be non-contiguous (for example 2, 3, 5, 6, 9);
    when explicit frame counts are supplied positionally, they are aligned to the observed
    fold IDs for that CSV rather than assuming folds 1..K

Note: `train_accuracy` / `test_accuracy` in source logs are balanced accuracy
(sklearn balanced_accuracy_score) when produced by current `util_features` /
SAR-RARP training code—not plain accuracy.

Outputs:
  - prints a human-readable summary table
  - optionally writes a summary CSV (default: ./gvr_logs/summary_gvr_best_by_test.csv)
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "test_f1",
    "test_accuracy",
    "test_jaccard",
]

# JIGSAWS: approximate number of test frames per LOSO fold (fold -> count).
DEFAULT_FOLD_FRAME_COUNTS: Dict[int, int] = {
    1: 8332,
    2: 6056,
    3: 7066,
    4: 6979,
    5: 5433,
}
DEFAULT_LOUO_FOLD_IDS: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 9)


@dataclass(frozen=True)
class RunKey:
    model: str
    prompt_type: str
    split_scheme: str
    learning_rate: float
    weight_decay: float


def _infer_run_key(df: pd.DataFrame, csv_path: Optional[str] = None) -> RunKey:
    model = str(df["model"].iloc[0])
    prompt_type = str(df["prompt_type"].iloc[0]) if "prompt_type" in df.columns else "unknown"
    split_scheme = "unknown"
    if "split_scheme" in df.columns:
        split_scheme = str(df["split_scheme"].iloc[0]).strip().lower()
    elif csv_path is not None:
        base = os.path.basename(csv_path).lower()
        if "_loso_" in base:
            split_scheme = "loso"
        elif "_louo_" in base:
            split_scheme = "louo"
    lr = float(df["learning_rate"].iloc[0]) if "learning_rate" in df.columns else float("nan")
    wd = float(df["weight_decay"].iloc[0]) if "weight_decay" in df.columns else float("nan")
    return RunKey(
        model=model,
        prompt_type=prompt_type,
        split_scheme=split_scheme,
        learning_rate=lr,
        weight_decay=wd,
    )


def _best_epoch_rows_by_fold(df: pd.DataFrame) -> pd.DataFrame:
    # We choose "best" epochs within each fold by the highest test_f1.
    # This matches the requirement: for each model/prompt setting, in each fold,
    # consider the epoch with the highest test F1.
    required = {"fold", "epoch", "test_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Ensure numeric
    df = df.copy()
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["test_f1"] = pd.to_numeric(df["test_f1"], errors="coerce")

    # Drop rows with NaNs in key columns
    df = df.dropna(subset=["fold", "epoch", "test_f1"])

    # For each fold, choose row with max test_f1; tie-breaker: earliest epoch (smaller epoch)
    df_sorted = df.sort_values(["fold", "test_f1", "epoch"], ascending=[True, False, True])
    best_rows = df_sorted.groupby("fold", as_index=False).head(1)
    return best_rows


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    # sample std (ddof=1) when we have >=2 folds, else 0
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return float(np.mean(arr)), std


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    """sum(w_i * x_i) / sum(w_i) over finite pairs with w_i > 0."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0 or w.size == 0 or v.size != w.size:
        return float("nan")
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.dot(v[mask], w[mask]) / np.sum(w[mask]))


def _canonical_fold_ids(split_scheme: str) -> Optional[Tuple[int, ...]]:
    normalized = str(split_scheme).strip().lower()
    if normalized == "loso":
        return tuple(sorted(DEFAULT_FOLD_FRAME_COUNTS))
    if normalized == "louo":
        return DEFAULT_LOUO_FOLD_IDS
    return None


def _parse_fold_frame_counts(
    s: Optional[str],
    observed_folds: Optional[List[int]] = None,
    split_scheme: str = "unknown",
) -> Optional[Dict[int, int]]:
    """
    - Empty / None: return None so the caller can choose scheme-specific defaults.
    - Comma-separated integers: counts aligned to the observed folds for the current CSV
      when lengths match, otherwise to the canonical fold order for the split scheme when
      lengths match, otherwise to folds 1, 2, ... in order.
    - Or per-fold '1:8332,2:6056,...'
    """
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return None
    if ":" in parts[0]:
        out: Dict[int, int] = {}
        for p in parts:
            fold_s, count_s = p.split(":", 1)
            out[int(fold_s.strip())] = int(count_s.strip())
        return out
    counts = [int(x) for x in parts]
    observed = sorted({int(f) for f in (observed_folds or [])})
    if observed and len(counts) == len(observed):
        return {fold: count for fold, count in zip(observed, counts)}
    canonical = _canonical_fold_ids(split_scheme)
    if canonical is not None and len(counts) == len(canonical):
        return {fold: count for fold, count in zip(canonical, counts)}
    return {i + 1: c for i, c in enumerate(counts)}


def _default_fold_frame_counts(folds: List[int], split_scheme: str) -> Dict[int, int]:
    observed = sorted({int(f) for f in folds})
    normalized = str(split_scheme).strip().lower()
    if normalized == "loso":
        return dict(DEFAULT_FOLD_FRAME_COUNTS)
    if normalized == "louo":
        return {fold: 1 for fold in observed}
    if all(fold in DEFAULT_FOLD_FRAME_COUNTS for fold in observed):
        return dict(DEFAULT_FOLD_FRAME_COUNTS)
    return {fold: 1 for fold in observed}


def _weights_for_folds(folds: List[int], fold_frame_counts: Dict[int, int]) -> List[float]:
    w: List[float] = []
    for f in folds:
        key = int(f)
        if key not in fold_frame_counts:
            raise ValueError(
                f"Fold {key} has no frame count; configured folds: {sorted(fold_frame_counts.keys())}"
            )
        w.append(float(fold_frame_counts[key]))
    return w


def summarize_csv(
    csv_path: str,
    metrics: List[str],
    fold_frame_counts_arg: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Backfill prompt_type / weight_decay columns if using older CSV variants
    if "prompt_type" not in df.columns:
        # Try to parse from filename: ..._{model}_{prompt_type}_lr_...
        base = os.path.basename(csv_path)
        parts = base.replace(".csv", "").split("_")
        # training_gvrerr_{model}_{prompt_type}_lr_{lr}_wd_{wd}
        if len(parts) >= 4:
            df["prompt_type"] = parts[3]
        else:
            df["prompt_type"] = "unknown"
    if "weight_decay" not in df.columns:
        df["weight_decay"] = np.nan

    best = _best_epoch_rows_by_fold(df)
    best = best.sort_values("fold")
    fold_list = [int(x) for x in best["fold"].tolist()]

    key = _infer_run_key(df, csv_path=csv_path)
    counts = _parse_fold_frame_counts(
        fold_frame_counts_arg,
        observed_folds=fold_list,
        split_scheme=key.split_scheme,
    )
    if counts is None:
        counts = _default_fold_frame_counts(fold_list, key.split_scheme)
    fold_weights = _weights_for_folds(fold_list, counts)

    out: Dict[str, object] = {
        "csv": os.path.basename(csv_path),
        "model": key.model,
        "prompt_type": key.prompt_type,
        "split_scheme": key.split_scheme,
        "learning_rate": key.learning_rate,
        "weight_decay": key.weight_decay,
        "num_folds": int(best["fold"].nunique()),
    }

    for m in metrics:
        if m not in best.columns:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
            out[f"{m}_weighted"] = float("nan")
            continue
        vals_series = pd.to_numeric(best[m], errors="coerce")
        vals = vals_series.tolist()
        mean, std = _mean_std([float(x) for x in vals if np.isfinite(x)])
        wmean = _weighted_mean(vals, fold_weights)
        out[f"{m}_mean"] = mean
        out[f"{m}_std"] = std
        out[f"{m}_weighted"] = wmean

    # Also capture selected epochs per fold (useful sanity check)
    fold_epochs = best.sort_values("fold")[["fold", "epoch", "test_f1"]].copy()
    out["best_epochs_by_fold"] = ";".join(
        [f"{int(r.fold)}:{int(r.epoch)}(test_f1={float(r.test_f1):.4f})" for r in fold_epochs.itertuples(index=False)]
    )

    return pd.DataFrame([out])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvr_logs_dir",
        type=str,
        default="./outputs/jigsaws/logs",
        help="Directory containing training_gvrerr_*.csv files",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="training_gvrerr_*.csv",
        help="Glob pattern within gvr_logs_dir (default: training_gvrerr_*.csv)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated metrics to summarize (default: {','.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="./outputs/jigsaws/logs/summary_gvr_best_by_test.csv",
        help="Where to write the summary CSV",
    )
    parser.add_argument(
        "--fold_frame_counts",
        type=str,
        default=None,
        help=(
            "Frame counts for weighted aggregation. Default: JIGSAWS LOSO fold counts "
            f"{DEFAULT_FOLD_FRAME_COUNTS}; LOUO defaults to equal per-fold weights unless "
            "counts are supplied. "
            'Pass comma-separated counts aligned to the observed folds for each CSV '
            '(e.g. folds 2,3,5,6,9 -> "1200,900,1100,1000,950"), or explicit '
            '"2:1200,3:900,5:1100,...".'
        ),
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    pattern = os.path.join(args.gvr_logs_dir, args.glob_pattern)
    csvs = sorted(glob.glob(pattern))
    if not csvs:
        raise SystemExit(f"[ERROR] No CSVs found matching: {pattern}")

    rows = []
    for csv_path in csvs:
        try:
            rows.append(
                summarize_csv(
                    csv_path,
                    metrics=metrics,
                    fold_frame_counts_arg=args.fold_frame_counts,
                )
            )
        except Exception as e:
            print(f"[WARN] Skipping {csv_path} due to error: {e}")

    if not rows:
        raise SystemExit("[ERROR] No CSVs could be summarized.")

    summary = pd.concat(rows, ignore_index=True)

    # For your primary request: one metric per (model,prompt_type,split_scheme).
    # Keep lr/wd as part of the key in case you have multiple sweeps, but we also provide a collapsed view.
    key_cols = ["model", "prompt_type", "split_scheme", "learning_rate", "weight_decay"]
    sort_cols = ["model", "prompt_type", "split_scheme", "learning_rate", "weight_decay"]
    summary = summary.sort_values(sort_cols)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)

    # Print a compact table focused on test metrics if present.
    display_cols = ["model", "prompt_type", "split_scheme", "num_folds"]
    if "test_f1_weighted" in summary.columns:
        display_cols += ["test_f1_weighted"]
    if "test_f1_mean" in summary.columns:
        display_cols += ["test_f1_mean", "test_f1_std"]
    if "test_accuracy_weighted" in summary.columns:
        display_cols += ["test_accuracy_weighted"]
    if "test_accuracy_mean" in summary.columns:
        display_cols += ["test_accuracy_mean", "test_accuracy_std"]
    if "test_jaccard_weighted" in summary.columns:
        display_cols += ["test_jaccard_weighted"]
    if "test_jaccard_mean" in summary.columns:
        display_cols += ["test_jaccard_mean", "test_jaccard_std"]
    display_cols += ["learning_rate", "weight_decay"]

    print("\n=== GVR summary (best epoch per fold by test_f1; weighted = frame-weighted across folds) ===")
    print(summary[display_cols].to_string(index=False))
    print(f"\n[INFO] Wrote summary CSV: {args.out_csv}")


if __name__ == "__main__":
    main()


