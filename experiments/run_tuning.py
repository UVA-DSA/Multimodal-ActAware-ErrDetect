#!/usr/bin/env python3
"""
Random-search hyperparameter tuning driver for the activity-aware error detection models.

Targets (model x dataset combinations from the paper):
  - jigsaws_prompt : Activity Prompting (Img+Txt) on JIGSAWS      -> jigsaws/train_eval_gvr_prompt_features.py
  - jigsaws_kin    : Activity Kinematics Fusion (Img+Txt+Kin)     -> jigsaws/train_eval_gvr_kin_features.py
  - sarrarp_prompt : Activity Prompting (Img+Txt) on SAR-RARP50   -> SAR_RARP/train_eval_gvr_error.py

Protocol:
  * every trial trains with a validation split (val videos held out from the training
    videos; the test set is never used for selection),
  * the objective is validation F1 at the best validation epoch (averaged over the
    JIGSAWS folds that were run, or over --repeats for SAR-RARP50),
  * test metrics at the selected epoch are recorded for reporting only.

The driver invokes the standard training scripts via subprocess, tags each run with a
unique --log_tag, parses the produced CSV logs, and appends one row per trial to
{outdir}/trials.csv. It is resumable: trials whose config hash already appears in
trials.csv are skipped.

Search spaces are JSON files (see experiments/search_spaces/*.json) with the shape:
  {
    "fixed":  {"--val_ratio": 0.2, "--split_seed": 0, ...},
    "search": {
      "--learning_rate": {"type": "loguniform", "low": 1e-5, "high": 2e-3},
      "--dropout":       {"type": "choice", "values": [0.1, 0.3, 0.5]},
      ...
    }
  }

Example (stage 1, reduced folds):
  python experiments/run_tuning.py --target jigsaws_prompt \
      --space experiments/search_spaces/jigsaws_prompt.json \
      --trials 30 --folds 1,3,5 --outdir outputs/tuning/jigsaws_prompt/stage1

Run from the repository root.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

TARGETS = {
    "jigsaws_prompt": {
        "script": os.path.join("jigsaws", "train_eval_gvr_prompt_features.py"),
        "log_dir": os.path.join("outputs", "jigsaws", "logs"),
        "log_glob": "training_gvrerr_*_{tag}.csv",
        "ckpt_glob": os.path.join("outputs", "jigsaws", "checkpoints", "gvrerr_*_{tag}_best.pth"),
        "kind": "jigsaws",
    },
    "jigsaws_kin": {
        "script": os.path.join("jigsaws", "train_eval_gvr_kin_features.py"),
        "log_dir": os.path.join("outputs", "jigsaws", "logs"),
        "log_glob": "training_gvrkin_*_{tag}.csv",
        "ckpt_glob": os.path.join("outputs", "jigsaws", "checkpoints", "gvrkin_*_{tag}_best.pth"),
        "kind": "jigsaws",
    },
    "sarrarp_prompt": {
        "script": os.path.join("SAR_RARP", "train_eval_gvr_error.py"),
        "log_dir": os.path.join("outputs", "sar_rarp50", "gvrerror_features"),
        "log_glob": "gvrerror_*_{tag}_metrics.csv",
        "ckpt_glob": os.path.join("outputs", "sar_rarp50", "gvrerror_features", "gvrerror_*_{tag}_best.pth"),
        "kind": "sarrarp",
    },
}


def sample_param(spec: Dict, rng: random.Random):
    ptype = spec.get("type", "choice")
    if ptype == "choice":
        return rng.choice(spec["values"])
    if ptype == "uniform":
        val = rng.uniform(float(spec["low"]), float(spec["high"]))
    elif ptype == "loguniform":
        val = math.exp(rng.uniform(math.log(float(spec["low"])), math.log(float(spec["high"]))))
    elif ptype == "const":
        return spec["value"]
    else:
        raise ValueError(f"Unknown sample type: {ptype}")
    if spec.get("cast") == "int":
        return int(round(val))
    return round(val, 8)


def config_hash(config: Dict) -> str:
    blob = json.dumps(config, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def build_command(python_exe: str, script: str, fixed: Dict, sampled: Dict, extra: Dict, tag: str) -> List[str]:
    cmd = [python_exe, script]
    merged = {}
    merged.update(fixed)
    merged.update(sampled)
    merged.update(extra)
    for k, v in merged.items():
        if v is None or v == "":
            continue
        cmd.extend([k, str(v)])
    cmd.extend(["--log_tag", tag])
    return cmd


def objective_series(df: pd.DataFrame, objective: str, f1_col: str, acc_col: str) -> pd.Series:
    """
    Per-epoch score used both to pick the epoch and to rank configurations.

    `f1` alone is gameable on these imbalanced window datasets: a near-constant
    "always error" predictor reaches F1 ~0.79 on JIGSAWS while its balanced accuracy
    sits at chance (0.5). `f1_bacc` averages F1 with balanced accuracy, so a
    degenerate predictor cannot win, and both metrics are ones the paper reports.
    """
    f1 = df[f1_col].astype(float)
    if objective == "f1":
        return f1
    bacc = df[acc_col].astype(float)
    if objective == "bacc":
        return bacc
    if objective == "f1_bacc":
        return 0.5 * (f1 + bacc)
    raise ValueError(f"Unknown objective: {objective!r}")


def parse_jigsaws_logs(log_dir: str, pattern: str, tag: str, objective: str = "f1_bacc") -> Optional[Dict]:
    hits = sorted(glob.glob(os.path.join(log_dir, pattern.format(tag=tag))))
    if not hits:
        return None
    df = pd.concat([pd.read_csv(p) for p in hits], ignore_index=True)
    if "val_f1" not in df.columns or df["val_f1"].isna().all():
        return None
    per_fold = []
    for fold, fdf in df.groupby("fold"):
        score = objective_series(fdf, objective, "val_f1", "val_accuracy")
        sel = fdf.loc[score.idxmax()]
        per_fold.append(
            {
                "fold": fold,
                "score": float(score.max()),
                "val_f1": float(sel["val_f1"]),
                "val_accuracy": float(sel.get("val_accuracy", float("nan"))),
                "val_jaccard": float(sel.get("val_jaccard", float("nan"))),
                "test_f1": float(sel["test_f1"]),
                "test_accuracy": float(sel["test_accuracy"]),
                "test_jaccard": float(sel["test_jaccard"]),
                "epoch": int(sel["epoch"]),
            }
        )
    pf = pd.DataFrame(per_fold)
    return {
        "objective": float(pf["score"].mean()),
        "objective_std": float(pf["score"].std(ddof=0)),
        "mean_val_f1": float(pf["val_f1"].mean()),
        "mean_val_acc": float(pf["val_accuracy"].mean()),
        "mean_test_f1_at_sel": float(pf["test_f1"].mean()),
        "mean_test_acc_at_sel": float(pf["test_accuracy"].mean()),
        "mean_test_jaccard_at_sel": float(pf["test_jaccard"].mean()),
        "n_folds": int(len(pf)),
        "mean_best_epoch": float(pf["epoch"].mean()),
    }


def parse_sarrarp_logs(log_dir: str, pattern: str, tag: str, objective: str = "f1_bacc") -> Optional[Dict]:
    hits = sorted(glob.glob(os.path.join(log_dir, pattern.format(tag=tag))))
    if not hits:
        return None
    scores, vals, vaccs, tests, accs, jacs, epochs = [], [], [], [], [], [], []
    for p in hits:
        df = pd.read_csv(p)
        if "val_f1" not in df.columns or df["val_f1"].isna().all():
            continue
        score = objective_series(df, objective, "val_f1", "val_acc")
        sel = df.loc[score.idxmax()]
        scores.append(float(score.max()))
        vals.append(float(sel["val_f1"]))
        vaccs.append(float(sel["val_acc"]))
        tests.append(float(sel["test_f1"]))
        accs.append(float(sel["test_acc"]))
        jacs.append(float(sel["test_jaccard"]))
        epochs.append(int(sel["epoch"]))
    if not scores:
        return None
    s = pd.Series(scores)
    return {
        "objective": float(s.mean()),
        "objective_std": float(s.std(ddof=0)),
        "mean_val_f1": float(pd.Series(vals).mean()),
        "mean_val_acc": float(pd.Series(vaccs).mean()),
        "mean_test_f1_at_sel": float(pd.Series(tests).mean()),
        "mean_test_acc_at_sel": float(pd.Series(accs).mean()),
        "mean_test_jaccard_at_sel": float(pd.Series(jacs).mean()),
        "n_folds": int(len(scores)),
        "mean_best_epoch": float(pd.Series(epochs).mean()),
    }


def cleanup_checkpoints(ckpt_glob: str, tag: str) -> int:
    """Delete the checkpoints a tuning trial wrote; searches keep only metrics."""
    removed = 0
    for path in glob.glob(ckpt_glob.format(tag=tag)):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def load_done_hashes(trials_csv: str) -> set:
    if not os.path.exists(trials_csv):
        return set()
    try:
        df = pd.read_csv(trials_csv)
        return set(df["config_hash"].astype(str).tolist())
    except Exception:
        return set()


def append_row(trials_csv: str, row: Dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(trials_csv)
    df.to_csv(trials_csv, mode="a", header=header, index=False)


def main():
    ap = argparse.ArgumentParser(description="Random-search tuning driver (validation-based).")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS.keys()))
    ap.add_argument("--space", required=True, help="Path to the JSON search-space file")
    ap.add_argument("--trials", type=int, default=30, help="Number of NEW trials to run")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Runs per config (SAR-RARP50: >1 averages over the run-to-run randomness)")
    ap.add_argument("--outdir", required=True, help="Directory for trials.csv, best_config.json, per-trial logs")
    ap.add_argument("--folds", type=str, default=None,
                    help="JIGSAWS only: comma-separated fold subset forwarded to the training script")
    ap.add_argument("--prompt_type", type=str, default=None,
                    help="Override the prompt type fixed in the space file")
    ap.add_argument("--num_epochs", type=int, default=None, help="Override epochs fixed in the space file")
    ap.add_argument("--sampler_seed", type=int, default=None,
                    help="Optional seed for the config sampler (pass explicitly for a reproducible trial list)")
    ap.add_argument("--tag_prefix", type=str, default="",
                    help="Prefix for per-trial --log_tag values. Give each parallel shard its own prefix "
                         "so concurrent workers never write the same log/checkpoint file.")
    ap.add_argument("--objective", type=str, default="f1_bacc", choices=["f1", "f1_bacc", "bacc"],
                    help="Validation score used to pick the epoch and rank configs. Default f1_bacc "
                         "(mean of F1 and balanced accuracy) because plain F1 is maximized by a "
                         "degenerate always-error predictor on these imbalanced window datasets.")
    ap.add_argument("--cleanup_checkpoints", type=int, default=1,
                    help="If 1 (default), delete each trial's checkpoints after parsing its metrics. "
                         "Searches only need the CSV metrics; checkpoints are ~40 MB per fold.")
    ap.add_argument("--max_hours", type=float, default=1e9, help="Stop launching new trials after this budget")
    ap.add_argument("--trial_timeout_min", type=float, default=180.0, help="Per-trial subprocess timeout")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    args = ap.parse_args()

    target = TARGETS[args.target]
    with open(args.space, encoding="utf-8") as f:
        space = json.load(f)
    fixed = dict(space.get("fixed", {}))
    search = space.get("search", {})

    extra = {}
    if args.folds:
        extra["--folds"] = args.folds
    if args.prompt_type:
        extra["--prompt_type"] = args.prompt_type
    if args.num_epochs is not None:
        extra["--num_epochs"] = args.num_epochs

    os.makedirs(args.outdir, exist_ok=True)
    logs_dir = os.path.join(args.outdir, "trial_stdout")
    os.makedirs(logs_dir, exist_ok=True)
    trials_csv = os.path.join(args.outdir, "trials.csv")
    done = load_done_hashes(trials_csv)

    rng = random.Random(args.sampler_seed)
    started = time.time()
    launched = 0
    attempts = 0
    while launched < args.trials and attempts < args.trials * 20:
        attempts += 1
        if (time.time() - started) / 3600.0 > args.max_hours:
            print(f"[INFO] Budget of {args.max_hours}h reached; stopping.")
            break
        sampled = {k: sample_param(v, rng) for k, v in search.items()}
        full_config = {**fixed, **sampled, **extra}
        chash = config_hash(full_config)
        if chash in done:
            continue
        done.add(chash)
        launched += 1

        run_metrics = []
        status = "ok"
        t0 = time.time()
        for rep in range(max(1, args.repeats)):
            tag = f"{args.tag_prefix}{chash}r{rep}" if args.repeats > 1 else f"{args.tag_prefix}{chash}"
            cmd = build_command(args.python, target["script"], fixed, sampled, extra, tag)
            print(f"\n[TRIAL {launched}/{args.trials}] {chash} rep {rep}:\n  {' '.join(cmd)}")
            if args.dry_run:
                continue
            stdout_path = os.path.join(logs_dir, f"{chash}_r{rep}.log")
            try:
                with open(stdout_path, "w", encoding="utf-8") as out:
                    subprocess.run(
                        cmd,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        timeout=args.trial_timeout_min * 60.0,
                        check=True,
                    )
            except subprocess.TimeoutExpired:
                status = "timeout"
                print(f"[WARN] trial {chash} rep {rep} timed out")
                continue
            except subprocess.CalledProcessError as e:
                status = f"failed({e.returncode})"
                print(f"[WARN] trial {chash} rep {rep} failed; see {stdout_path}")
                continue

            if target["kind"] == "jigsaws":
                m = parse_jigsaws_logs(target["log_dir"], target["log_glob"], tag, args.objective)
            else:
                m = parse_sarrarp_logs(target["log_dir"], target["log_glob"], tag, args.objective)
            if m is not None:
                run_metrics.append(m)
            else:
                status = "no_logs"
                print(f"[WARN] no parsable logs for trial {chash} rep {rep}")

            if args.cleanup_checkpoints:
                cleanup_checkpoints(target["ckpt_glob"], tag)

        if args.dry_run:
            continue

        row = {"trial": launched, "config_hash": chash, "status": status,
               "objective_name": args.objective,
               "elapsed_sec": round(time.time() - t0, 1)}
        row.update({k.lstrip("-"): v for k, v in {**sampled, **extra}.items()})
        if run_metrics:
            mdf = pd.DataFrame(run_metrics)
            row.update(
                {
                    "objective": float(mdf["objective"].mean()),
                    "objective_std": float(mdf["objective"].std(ddof=0)) if len(mdf) > 1 else float(mdf["objective_std"].mean()),
                    "mean_test_f1_at_sel": float(mdf["mean_test_f1_at_sel"].mean()),
                    "mean_test_acc_at_sel": float(mdf["mean_test_acc_at_sel"].mean()),
                    "mean_test_jaccard_at_sel": float(mdf["mean_test_jaccard_at_sel"].mean()),
                    "mean_val_f1": float(mdf["mean_val_f1"].mean()),
                    "mean_val_acc": float(mdf["mean_val_acc"].mean()),
                    "mean_best_epoch": float(mdf["mean_best_epoch"].mean()),
                    "n_runs": int(len(mdf)),
                }
            )
        append_row(trials_csv, row)

    if args.dry_run:
        return

    # Rank and export the best config
    if os.path.exists(trials_csv):
        df = pd.read_csv(trials_csv)
        ok = df[df["status"] == "ok"].dropna(subset=["objective"]) if "objective" in df.columns else pd.DataFrame()
        if len(ok):
            ok = ok.sort_values("objective", ascending=False)
            best = ok.iloc[0].to_dict()
            with open(os.path.join(args.outdir, "best_config.json"), "w", encoding="utf-8") as f:
                json.dump({"target": args.target, "best": best}, f, indent=2, default=str)
            ok.head(10).to_csv(os.path.join(args.outdir, "top10.csv"), index=False)
            print("\n[INFO] Top 5 by validation objective:")
            cols = [c for c in ("config_hash", "objective", "objective_std", "mean_test_f1_at_sel") if c in ok.columns]
            print(ok.head(5)[cols].to_string(index=False))
        else:
            print("[WARN] No successful trials recorded.")


if __name__ == "__main__":
    main()
