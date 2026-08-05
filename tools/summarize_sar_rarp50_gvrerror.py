#!/usr/bin/env python3
"""
Summarize SAR_RARP50 GVR error runs from epoch-level CSV logs.

Expected input CSVs (written by SAR_RARP/train_eval_gvr_error.py):
  - Older format:
      gvrerror_{feature_tag}_{prompt_type}_lr{...}_wd{...}_bs{...}_seed{...}_metrics.csv
  - Newer format (includes an architecture tag between prompt_type and lr):
      gvrerror_{feature_tag}_{prompt_type}_{arch_tag}_lr{...}_wd{...}_bs{...}_seed{...}_metrics.csv

This script:
- parses (feature_tag, prompt_type, lr, wd, bs, seed) from filenames
- for each CSV, selects the epoch row with the highest `test_f1` (tie-break: smallest epoch)
- optionally collapses multiple runs per (feature_tag,prompt_type) to keep only the best run by `best_test_f1`
- writes one aggregated CSV
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


PROMPT_TYPES_DEFAULT = [
    "error",
    "gesture",
    "context",
    "gesture_error",
    "lowlevel_gesture_error",
]


@dataclass(frozen=True)
class ParsedName:
    csv: str
    feature_tag: str
    prompt_type: str
    arch_tag: str
    lr: Optional[float]
    wd: Optional[float]
    bs: Optional[int]
    seed: Optional[int]
    trstep: Optional[int]
    testep: Optional[int]
    nopool: Optional[int]


_RUN_RE = re.compile(
    r"^gvrerror_(?P<body>.+?)"
    r"_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)_bs(?P<bs>\d+)_seed(?P<seed>\d+)"
    r"(?P<extra>(?:_[^_]+)*)_metrics\.csv$"
)


def _split_feature_prompt_arch(body: str, prompt_types: List[str]) -> Tuple[str, str, str]:
    """
    Split `body` into:
      feature_tag + '_' + prompt_type [+ '_' + arch_tag]
    by finding the LAST prompt-type token occurrence.
    """
    candidates: List[Tuple[int, str]] = []
    for p in sorted(prompt_types, key=len, reverse=True):
        token = f"_{p}"
        start = 0
        while True:
            idx = body.find(token, start)
            if idx < 0:
                break
            end_idx = idx + len(token)
            # prompt token must be followed by "_" (arch exists) or end-of-string
            if end_idx == len(body) or body[end_idx] == "_":
                candidates.append((idx, p))
            start = idx + 1
    if not candidates:
        raise ValueError(f"Could not locate prompt_type token in body: {body}")

    # Pick the longest matching prompt token first to avoid splitting
    # "gesture_error" into prompt="error" with feature suffix "..._gesture".
    # Tie-break by right-most occurrence for repeated same prompt token.
    idx, prompt_type = max(candidates, key=lambda x: (len(x[1]), x[0]))
    feature_tag = body[:idx]
    tail = body[idx + 1 + len(prompt_type):]  # strip "_{prompt_type}"
    arch_tag = tail[1:] if tail.startswith("_") else ""

    if not feature_tag:
        raise ValueError(f"Empty feature_tag parsed from body: {body}")
    return feature_tag, prompt_type, arch_tag


def _parse_filename(path: str, prompt_types: List[str]) -> ParsedName:
    base = os.path.basename(path)
    if not base.startswith("gvrerror_") or not base.endswith("_metrics.csv"):
        raise ValueError(f"Unexpected filename format: {base}")

    m = _RUN_RE.match(base)
    if m is None:
        raise ValueError(f"Could not parse run suffix from: {base}")

    body = str(m.group("body"))
    feature_tag, prompt_type, arch_tag = _split_feature_prompt_arch(body, prompt_types)

    lr = wd = None
    bs = seed = None
    # Extract numeric run settings from matched regex groups
    try:
        lr = float(m.group("lr"))
    except Exception:
        lr = None
    try:
        wd = float(m.group("wd"))
    except Exception:
        wd = None
    try:
        bs = int(m.group("bs"))
    except Exception:
        bs = None
    try:
        seed = int(m.group("seed"))
    except Exception:
        seed = None

    trstep = testep = nopool = None
    extra = str(m.groupdict().get("extra", "") or "")
    if extra:
        for tok in extra.split("_"):
            if not tok:
                continue
            if tok.startswith("trstep"):
                try:
                    trstep = int(tok[len("trstep"):])
                except Exception:
                    pass
            elif tok.startswith("testep"):
                try:
                    testep = int(tok[len("testep"):])
                except Exception:
                    pass
            elif tok.startswith("nopool"):
                try:
                    nopool = int(tok[len("nopool"):])
                except Exception:
                    pass

    return ParsedName(
        csv=base,
        feature_tag=feature_tag,
        prompt_type=prompt_type,
        arch_tag=arch_tag,
        lr=lr,
        wd=wd,
        bs=bs,
        seed=seed,
        trstep=trstep,
        testep=testep,
        nopool=nopool,
    )


def _best_epoch_row(df: pd.DataFrame) -> pd.Series:
    if "epoch" not in df.columns or "test_f1" not in df.columns:
        raise ValueError("CSV must contain columns: epoch, test_f1")

    d = df.copy()
    d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
    d["test_f1"] = pd.to_numeric(d["test_f1"], errors="coerce")
    d = d.dropna(subset=["epoch", "test_f1"])
    if d.empty:
        raise ValueError("CSV has no valid rows after numeric coercion for (epoch,test_f1)")

    # tie-break: earliest epoch
    d = d.sort_values(["test_f1", "epoch"], ascending=[False, True])
    return d.iloc[0]


def summarize_one(csv_path: str, prompt_types: List[str]) -> Dict[str, object]:
    meta = _parse_filename(csv_path, prompt_types=prompt_types)
    df = pd.read_csv(csv_path)
    best = _best_epoch_row(df)

    out: Dict[str, object] = {
        "csv": meta.csv,
        "feature_tag": meta.feature_tag,
        "prompt_type": meta.prompt_type,
        "arch_tag": meta.arch_tag,
        "run_nopool": meta.nopool,
        "best_test_f1": float(best["test_f1"]),
    }

    # Carry through other common metrics if present
    for k in [
        "val_f1",
        "train_f1",
        "test_acc",
        "val_acc",
        "train_acc",
        "test_accuracy",
        "test_jaccard",
        "val_accuracy",
        "val_jaccard",
        "train_loss",
    ]:
        if k in best.index:
            out[f"best_{k}"] = best[k]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize SAR_RARP50 gvrerror_features runs")
    p.add_argument(
        "--input_dir",
        type=str,
        default="./outputs/sar_rarp50/gvrerror_features",
        help="Directory containing gvrerror_*_metrics.csv files",
    )
    p.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="gvrerror_*_metrics.csv",
        help="Glob within input_dir",
    )
    p.add_argument(
        "--prompt_types",
        type=str,
        default=",".join(PROMPT_TYPES_DEFAULT),
        help="Comma-separated prompt types to help filename parsing",
    )
    p.add_argument(
        "--collapse_runs",
        action="store_true",
        help="If set, keep only the best run per (feature_tag,prompt_type) by best_test_f1",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="./error_detection_SAR_RARP50/gvrerror_features/summary_best_test_f1.csv",
        help="Where to write aggregated CSV",
    )
    args = p.parse_args()

    prompt_types = [x.strip() for x in args.prompt_types.split(",") if x.strip()]
    pattern = os.path.join(args.input_dir, args.glob_pattern)
    csvs = sorted(glob.glob(pattern))
    if not csvs:
        raise SystemExit(f"[ERROR] No CSVs found matching: {pattern}")

    rows: List[Dict[str, object]] = []
    for fp in csvs:
        try:
            rows.append(summarize_one(fp, prompt_types=prompt_types))
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not rows:
        raise SystemExit("[ERROR] No CSVs could be summarized.")

    out = pd.DataFrame(rows)
    # Prefer deterministic ordering; arch_tag is informative for newer runs.
    out = out.sort_values(["feature_tag", "prompt_type", "arch_tag", "best_test_f1"], ascending=[True, True, True, False])

    if args.collapse_runs:
        # Collapse across arch_tag/lr/wd/seed: keep best run per (feature_tag,prompt_type)
        out = out.sort_values(["feature_tag", "prompt_type", "best_test_f1", "csv"], ascending=[True, True, False, True])
        out = out.groupby(["feature_tag", "prompt_type"], as_index=False).head(1)
        out = out.sort_values(["feature_tag", "prompt_type"])

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[INFO] Wrote summary CSV: {args.out_csv}")


if __name__ == "__main__":
    main()


