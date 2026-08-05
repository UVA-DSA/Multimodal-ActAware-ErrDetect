#!/usr/bin/env python3
"""
Cross training script that uses ONLY precomputed features:
- Base visual features: extracted by extract_features.py (resnet/clip)
- Gesture prompt features: extracted by extract_gesture_prompt_features.py (finetuned gesture model hidden)

This replaces the CNNs inside `GVRModuleContexPred` with feature tensors loaded from disk.
It also integrates training robustness patterns from train_eval_gvr_error_features.py:
- argparse config
- AdamW + weight decay
- ReduceLROnPlateau scheduler
- early stopping
- optional pos_weight
"""

from __future__ import annotations

import argparse
import copy
import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from dataloader_dual_features import DualFeaturePaths, JIGSAWS_Gesture_DualFeatures, JIGSAWS_DualFeatures_SegmentWrapper, collate_fn_dual_features_context, split_selector
from dataloader_features import MODEL_FEATURE_DIMS
from jigsaws_splits import get_split_fold_ids, make_dataset_variant
from model_gvr_features import GVRModuleContexPredDualFeatures
from util_features import compute_pos_weight_context, train_context_dual_features, test_context_dual_features


def _infer_gesture_dim(gesture_root: str, task: str) -> int:
    # Find a sample features.pt
    for v in os.listdir(os.path.join(gesture_root, task)):
        fp = os.path.join(gesture_root, task, v, "features.pt")
        if os.path.exists(fp):
            d = torch.load(fp, map_location="cpu")
            if len(d) == 0:
                continue
            return int(next(iter(d.values())).shape[0])
    raise FileNotFoundError(f"Could not infer gesture feature dim from {gesture_root}/{task}/*/features.pt")


def main():
    parser = argparse.ArgumentParser(description="Train cross model using precomputed base+gesture features")
    parser.add_argument("--base_model", type=str, default="resnet50", choices=list(MODEL_FEATURE_DIMS.keys()))
    parser.add_argument("--gesture_feature_root", type=str, required=True, help="Root produced by extract_gesture_prompt_features.py (typically data/gesture_prompt_features/{ckpt_stem})")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--use_pos_weight", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--debug_eval", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--segment_length", type=int, default=40, help="Window length for sliding segments")
    parser.add_argument("--step_size", type=int, default=6, help="Step size for sliding windows")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed. Unset leaves RNGs unseeded.")
    parser.add_argument("--split_scheme", type=str, default="loso", choices=["loso", "louo"])
    parser.add_argument("--split_root", type=str, default=None,
                        help="Optional root containing split CSVs; defaults to ./LOSO or ./LOUO from --split_scheme")
    parser.add_argument(
        "--window_label_rule",
        type=str,
        default="majority",
        choices=["majority", "any_error"],
        help="How to derive one label per window from per-frame labels: majority (tie->error) or any_error (1 if any valid frame is error).",
    )
    args = parser.parse_args()

    args.position = True if args.position == 1 else False
    args.use_pos_weight = True if args.use_pos_weight == 1 else False
    args.debug_eval = True if args.debug_eval == 1 else False

    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.seed is not None:
        _set_seed(args.seed)

    device = torch.device(args.device)
    base_dim = MODEL_FEATURE_DIMS[args.base_model]

    # infer gesture dim from extracted prompt features
    ges_dim_s = _infer_gesture_dim(args.gesture_feature_root, "Suturing")
    ges_dim_n = _infer_gesture_dim(args.gesture_feature_root, "Needle_Passing")
    if ges_dim_s != ges_dim_n:
        print(f"[WARN] gesture dim mismatch: Suturing={ges_dim_s} Needle_Passing={ges_dim_n}; using Suturing dim")
    ges_dim = ges_dim_s
    d_embed = ges_dim  # align with original code's use of gesture branch dimension

    log_dir = "./outputs/jigsaws/logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(
        log_dir,
        f"training_crosscontext_{args.base_model}_{args.split_scheme}_gesdim{ges_dim}_lr_{args.learning_rate}.csv",
    )
    all_fold_logs = []

    print(f"[INFO] Split scheme: {args.split_scheme}")
    print(f"[INFO] Split root: {args.split_root or '(auto)'}")
    print(f"[INFO] Window label rule: {args.window_label_rule}")

    for fold in get_split_fold_ids(args.split_scheme):
        start_time = time.time()
        dataset_variant_s = make_dataset_variant("Suturing", args.split_scheme, fold)
        dataset_variant_n = make_dataset_variant("Needle_Passing", args.split_scheme, fold)

        split_s = split_selector(".", dataset_variant_s, split_root=args.split_root)
        split_n = split_selector(".", dataset_variant_n, split_root=args.split_root)

        paths = DualFeaturePaths(data_root=args.data_root, base_model=args.base_model, gesture_root=args.gesture_feature_root)

        def _build_sets(task: str, records):
            sets = []
            for fn in records:
                ds = JIGSAWS_Gesture_DualFeatures(fn if fn.endswith(".csv") else f"{fn}.csv", task, paths)
                ds = JIGSAWS_DualFeatures_SegmentWrapper(
                    ds, segment_length=args.segment_length, step_size=args.step_size
                )
                sets.append(ds)
            return sets

        train_sets = _build_sets("Suturing", split_s["train"]) + _build_sets("Needle_Passing", split_n["train"])
        test_sets = _build_sets("Suturing", split_s["test"]) + _build_sets("Needle_Passing", split_n["test"])

        train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dual_features_context)
        test_loader = DataLoader(ConcatDataset(test_sets), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_dual_features_context)

        model = GVRModuleContexPredDualFeatures(
            base_dim=base_dim,
            ges_dim=ges_dim,
            d_embed=d_embed,
            num_heads=args.num_heads,
            segment_length=args.segment_length,
            position=args.position,
            dropout=args.dropout,
            layer_norm1=False,
            layer_norm2=True,
            layer_norm3=False,
        ).to(device)

        if args.use_pos_weight:
            _, criterion = compute_pos_weight_context(
                train_loader, device, window_label_rule=args.window_label_rule
            )
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = None

        best_test_f1 = 0.0
        best_state = None
        epochs_wo = 0

        fold_log = {
            "fold": [],
            "epoch": [],
            "base_model": [],
            "split_scheme": [],
            "ges_dim": [],
            "lr": [],
            "train_f1": [],
            "train_acc": [],
            "train_jaccard": [],
            "test_f1": [],
            "test_acc": [],
            "test_jaccard": [],
        }

        print(f"\n{'='*60}")
        print(
            f"Fold {fold} [{args.split_scheme}] | "
            f"base_model={args.base_model} base_dim={base_dim} | gesture_dim={ges_dim}"
        )
        print(f"{'='*60}")

        for epoch in range(args.num_epochs):
            lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch+1}/{args.num_epochs} (LR={lr:.6f})")

            train_f1, train_acc, train_j = train_context_dual_features(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                max_grad_norm=args.max_grad_norm,
                window_label_rule=args.window_label_rule,
            )
            test_f1, test_acc, test_j = test_context_dual_features(
                model,
                test_loader,
                device,
                debug=False,
                criterion=criterion,
                window_label_rule=args.window_label_rule,
            )

            fold_log["fold"].append(fold)
            fold_log["epoch"].append(epoch + 1)
            fold_log["base_model"].append(args.base_model)
            fold_log["split_scheme"].append(args.split_scheme)
            fold_log["ges_dim"].append(ges_dim)
            fold_log["lr"].append(lr)
            fold_log["train_f1"].append(train_f1)
            fold_log["train_acc"].append(train_acc)
            fold_log["train_jaccard"].append(train_j)
            fold_log["test_f1"].append(test_f1)
            fold_log["test_acc"].append(test_acc)
            fold_log["test_jaccard"].append(test_j)

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_state = copy.deepcopy(model.state_dict())
                epochs_wo = 0
                print(f"  *** New best test_f1={best_test_f1:.4f} ***")
            else:
                epochs_wo += 1
                print(f"  No test improvement for {epochs_wo} epoch(s)")
                if epochs_wo >= args.patience:
                    print("  Early stopping")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Save fold checkpoint
        save_dir = "./outputs/jigsaws/checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"crosscontext_{args.base_model}_{args.split_scheme}_gesdim{ges_dim}_fold{fold}_best.pth",
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "base_model": args.base_model,
                "base_dim": base_dim,
                "gesture_feature_root": args.gesture_feature_root,
                "gesture_dim": ges_dim,
                "best_test_f1": best_test_f1,
                "args": vars(args),
            },
            save_path,
        )
        print(f"[INFO] Saved best fold model: {save_path}")

        all_fold_logs.append(pd.DataFrame(fold_log))
        print(f"[INFO] Fold {fold} took {(time.time() - start_time)/3600:.2f} hours")

    if all_fold_logs:
        pd.concat(all_fold_logs, ignore_index=True).to_csv(csv_path, index=False)
        print(f"[INFO] Logs saved to {csv_path}")

        # Summary
        best_vals = [df["test_f1"].max() for df in all_fold_logs]
        print("\n" + "=" * 60)
        print("SUMMARY - Best TEST F1 per fold")
        for fold_df in all_fold_logs:
            fold_id = int(fold_df["fold"].iloc[0])
            print(f"  Fold {fold_id}: {float(fold_df['test_f1'].max()):.4f}")
        print(f"  Avg: {float(np.mean(best_vals)):.4f}")


if __name__ == "__main__":
    main()


