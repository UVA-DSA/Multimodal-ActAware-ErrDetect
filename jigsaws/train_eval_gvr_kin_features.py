"""
Training script for GVR Error Detection using:
  - pre-extracted visual features (ResNet/CLIP) AND
  - kinematic signals

This script is the kinematics analogue of `train_eval_gvr_error_features.py`:
it mirrors the same training enhancements:
- AdamW + weight decay
- optional class-weighted loss (pos_weight)
- early stopping
- best-by-test checkpointing
- richer CSV logging (train/test)

Run feature extraction first:
  python extract_features.py --model resnet50
  python extract_features.py --model clip-vit-base-patch16
"""

import argparse
import copy
import os
import time
import random
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from dataloader_features import MODEL_FEATURE_DIMS, infer_feature_dim_from_root
from dataloader_kin_features import JIGSAWSDataKinFeatures, collate_fn_kin_features
from jigsaws_splits import get_split_fold_ids, make_dataset_variant
from model_gvr_features import GVRModulePredKinFeatures
from textprompts import (
    error_prompt,
    gesture_error_prompt,
    gesture_prompt,
    lowlevel_gesture_error_prompt,
    Context_Prompt,
)
from util_features import build_criterion
from util_kin_features import train_kin_features, test_kin_features, compute_pos_weight_kin_features


AVAILABLE_MODELS = list(MODEL_FEATURE_DIMS.keys())

PROMPT_SETS = {
    "error": error_prompt,
    "gesture_error": gesture_error_prompt,
    "gesture": gesture_prompt,
    "lowlevel_gesture_error": lowlevel_gesture_error_prompt,
    "context": Context_Prompt,
}
AVAILABLE_PROMPT_TYPES = list(PROMPT_SETS.keys())


parser = argparse.ArgumentParser(
    description="Train GVRModulePredKinFeatures model with pre-extracted features + kinematics.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Train with ResNet50 features + kinematics
  python train_eval_gvr_kin_features.py --learning_rate 0.0005 --model resnet50 --prompt_type lowlevel_gesture_error

  # Train with CLIP ViT-B/16 features + kinematics
  python train_eval_gvr_kin_features.py --learning_rate 0.0005 --model clip-vit-base-patch16 --prompt_type lowlevel_gesture_error
""",
)
parser.add_argument("--position", type=int, default=1, help="Use positional encoding if 1")
parser.add_argument("--learning_rate", type=float, default=0.00017561, help="Learning rate for tuning")
parser.add_argument("--model", type=str, default="resnet50", choices=AVAILABLE_MODELS,
                    help=f"Default feature extractor model. Options: {AVAILABLE_MODELS}")
parser.add_argument(
    "--feature_root",
    type=str,
    default=None,
    help="Optional custom feature root shaped as {root}/{task}/{video}/features.pt. "
         "Use this to train on gesture-finetuned or error-finetuned extracted features.",
)
parser.add_argument(
    "--feature_tag",
    type=str,
    default=None,
    help="Optional short tag for logs/checkpoints when --feature_root is used. Defaults to the feature_root basename.",
)
parser.add_argument("--prompt_type", type=str, default="gesture_error", choices=AVAILABLE_PROMPT_TYPES,
                    help=f"Prompt set to use. Options: {AVAILABLE_PROMPT_TYPES}")
parser.add_argument("--weight_decay", type=float, default=8.45e-06, help="Weight decay (L2 regularization)")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
parser.add_argument("--use_pos_weight", type=int, default=1, help="Use class-weighted loss if 1")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads in the model")
parser.add_argument(
    "--kin_downsample",
    type=int,
    default=3,
    help="Downsample factor for kinematics. Use 3 to align 30 Hz kinematics with 10 Hz visual frames numbered 0,3,6,...; "
         "kinematics are matched to kept frames by original frame number, so this can stay 3 with any --frame_subsample.",
)
parser.add_argument(
    "--frame_subsample",
    type=int,
    default=2,
    help="Keep every k-th stored frame. Stored JIGSAWS frames are 10 Hz; the default 2 gives the paper's 5 Hz rate.",
)
parser.add_argument("--debug_eval", type=int, default=0, help="If 1, print one eval batch's shape stats per epoch")
parser.add_argument("--use_pooled_branch", type=int, default=0, help="Use pooled features in final prediction if 1")
parser.add_argument("--segment_length", type=int, default=20, help="Window length in samples for sliding segments")
parser.add_argument("--train_step_size", type=int, default=10, help="Step size for training sliding windows")
parser.add_argument("--test_step_size", type=int, default=10, help="Step size for test sliding windows")
parser.add_argument("--seed", type=int, default=None,
                    help="Optional random seed. Unset (default) leaves all RNGs unseeded.")
parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.2,
    help="Fraction of training videos (per task) held out as a validation set. "
         "When > 0, early stopping and model selection use validation F1; when 0, the legacy "
         "behavior (selection on test F1) is kept.",
)
parser.add_argument(
    "--split_seed",
    type=int,
    default=0,
    help="Seed controlling the train/validation video split only. Pass an explicit value to keep "
         "the same validation videos across runs (e.g. during hyperparameter tuning).",
)
parser.add_argument("--folds", type=str, default=None,
                    help="Optional comma-separated subset of fold ids to run (e.g. '1,3,5'). Default: all folds.")
parser.add_argument("--selection_metric", type=str, default="f1_bacc",
                    choices=["f1", "f1_bacc", "bacc"],
                    help="Validation score used to choose the best epoch and drive early stopping. "
                         "Default f1_bacc (mean of F1 and balanced accuracy), matching the tuning campaign; "
                         "plain f1 selects positive-leaning epochs on these imbalanced datasets.")
parser.add_argument("--log_tag", type=str, default="",
                    help="Optional tag appended to log/checkpoint filenames (used by the tuning driver).")
parser.add_argument("--split_scheme", type=str, default="loso", choices=["loso", "louo"],
                    help="Cross-validation scheme: loso (trial-out) or louo (subject-out)")
parser.add_argument("--split_root", type=str, default=None,
                    help="Optional root containing split CSVs; defaults to ./LOSO or ./LOUO from --split_scheme")
parser.add_argument(
    "--window_label_rule",
    type=str,
    default="any_error",
    choices=["majority", "any_error"],
    help="How to derive one label per window from per-frame labels: majority (tie->error) or any_error (1 if any valid frame is error).",
)
# --- architectural variants (see model_gvr_features.py) ---
parser.add_argument("--fusion", type=str, default="gated", choices=["concat", "gated", "film"],
                    help="How the pooled summary is fused with the refined prompt features")
parser.add_argument("--pooled_mode", type=str, default="mean", choices=["tcn", "mean", "attn"],
                    help="Pooling used for the direct video+kinematics summary branch")
parser.add_argument("--attn_temperature", type=float, default=1.0,
                    help="Temperature for the prompt-to-frame attention (values < 1 sharpen attention)")
parser.add_argument("--kin_fusion", type=str, default="concat", choices=["concat", "gated"],
                    help="How per-frame kinematic features are combined with visual features")
parser.add_argument("--kin_dim_ratio", type=int, default=8,
                    help="Kinematics projection dim = d_text // kin_dim_ratio (paper uses 8)")
# --- loss variants ---
parser.add_argument("--loss", type=str, default="wbce", choices=["wbce", "focal"],
                    help="Training loss: class-weighted BCE (paper) or focal loss")
parser.add_argument("--focal_gamma", type=float, default=3.0, help="Focal loss gamma (only used with --loss focal)")
parser.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Optional label smoothing epsilon applied to binary targets")

args = parser.parse_args()


def _selection_score(f1, bal_acc, metric):
    """
    Score used to pick the best epoch (and drive early stopping).

    Plain F1 is gameable on these imbalanced window datasets: a near-constant
    "always error" predictor scores high F1 at chance-level balanced accuracy, so
    selecting on F1 alone lands on a positive-leaning epoch. `f1_bacc` averages F1
    with balanced accuracy and is the criterion the tuning campaign used, so it is
    the default here as well.
    """
    if metric == "f1":
        return f1
    if metric == "bacc":
        return bal_acc
    if metric == "f1_bacc":
        return 0.5 * (f1 + bal_acc)
    raise ValueError(f"Unknown selection_metric: {metric!r}")

args.position = True if args.position == 1 else False
args.use_pos_weight = True if args.use_pos_weight == 1 else False
args.debug_eval = True if args.debug_eval == 1 else False
args.use_pooled_branch = True if args.use_pooled_branch == 1 else False


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


def _resolve_feature_tag(model_name: str, feature_root: Optional[str], feature_tag: Optional[str]) -> str:
    if feature_tag:
        raw = feature_tag
    elif feature_root:
        raw = os.path.basename(os.path.normpath(feature_root)) or "custom_features"
    else:
        raw = model_name
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dim = infer_feature_dim_from_root(args.feature_root) if args.feature_root else MODEL_FEATURE_DIMS[args.model]
feature_tag = _resolve_feature_tag(args.model, args.feature_root, args.feature_tag)
feature_source_desc = args.feature_root or os.path.join("./data", "vid_features", args.model)
if args.feature_root and feature_dim != MODEL_FEATURE_DIMS[args.model]:
    print(
        f"[WARN] Inferred feature dim {feature_dim} from --feature_root, "
        f"which differs from {args.model} default dim {MODEL_FEATURE_DIMS[args.model]}."
    )

print(f"[INFO] Feature source tag: {feature_tag}")
print(f"[INFO] Feature extractor model: {args.model}")
print(f"[INFO] Feature root: {feature_source_desc}")
print(f"[INFO] Feature dimension: {feature_dim}")
print(f"[INFO] Prompt type: {args.prompt_type} (num_prompts={len(PROMPT_SETS[args.prompt_type])})")
print(f"[INFO] Using device: {device}")
print(f"[INFO] Learning rate: {args.learning_rate}")
print(f"[INFO] Weight decay: {args.weight_decay}")
print(f"[INFO] Dropout: {args.dropout}")
print(f"[INFO] Num heads: {args.num_heads}")
print(f"[INFO] Max grad norm: {args.max_grad_norm}")
print(f"[INFO] Early stopping patience: {args.patience}")
print(f"[INFO] Use pos_weight: {args.use_pos_weight}")
print(f"[INFO] Use pooled branch: {args.use_pooled_branch}")
print(f"[INFO] Kinematics downsample: {args.kin_downsample}")
print(f"[INFO] Segment length: {args.segment_length}")
print(f"[INFO] Train step size: {args.train_step_size}")
print(f"[INFO] Test step size: {args.test_step_size}")
print(f"[INFO] Seed: {args.seed}")
print(f"[INFO] Split scheme: {args.split_scheme}")
print(f"[INFO] Split root: {args.split_root or '(auto)'}")
print(f"[INFO] Window label rule: {args.window_label_rule}")


LOG_DIR = "./outputs/jigsaws/logs"
CKPT_DIR = "./outputs/jigsaws/checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
tag_suffix = f"_{args.log_tag}" if args.log_tag else ""
csv_filename = (
    f"training_gvrkin_{feature_tag}_{args.prompt_type}_{args.split_scheme}_"
    f"lr_{args.learning_rate}_wd_{args.weight_decay}{tag_suffix}.csv"
)
csv_path = os.path.join(LOG_DIR, csv_filename)


def _split_train_val_videos(dataset_list, val_ratio: float, rng: random.Random):
    """Split a list of per-video datasets into train/val lists (at least one video kept in train)."""
    if val_ratio <= 0 or len(dataset_list) < 2:
        return dataset_list, []
    shuffled = list(dataset_list)
    rng.shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * val_ratio))))
    return shuffled[n_val:], shuffled[:n_val]


all_fold_logs = []

fold_ids = list(get_split_fold_ids(args.split_scheme))
if args.folds:
    requested = {int(x) for x in args.folds.split(",") if x.strip()}
    fold_ids = [f for f in fold_ids if f in requested]
    print(f"[INFO] Restricting to folds: {fold_ids}")
for fold in fold_ids:
    start_time = time.time()
    dataset_variant_suturing = make_dataset_variant("Suturing", args.split_scheme, fold)
    dataset_variant_needle = make_dataset_variant("Needle_Passing", args.split_scheme, fold)

    suturing_data = JIGSAWSDataKinFeatures(
        dataset_variant_suturing,
        model_name=args.model,
        split_root=args.split_root,
        feature_root=args.feature_root,
        kin_downsample=args.kin_downsample,
        frame_subsample=args.frame_subsample,
    )
    needle_data = JIGSAWSDataKinFeatures(
        dataset_variant_needle,
        model_name=args.model,
        split_root=args.split_root,
        feature_root=args.feature_root,
        kin_downsample=args.kin_downsample,
        frame_subsample=args.frame_subsample,
    )

    # Optional validation split, carved from the training videos of each task.
    split_rng = random.Random(args.split_seed) if args.split_seed is not None else random.Random()
    sut_train, sut_val = _split_train_val_videos(suturing_data.train_dataset, args.val_ratio, split_rng)
    np_train, np_val = _split_train_val_videos(needle_data.train_dataset, args.val_ratio, split_rng)

    def _rewrap(ds_list, step_size):
        return [
            type(ds)(ds.original_dataset, segment_length=args.segment_length, step_size=step_size)
            for ds in ds_list
        ]

    train_datasets = _rewrap(sut_train, args.train_step_size) + _rewrap(np_train, args.train_step_size)
    val_datasets = _rewrap(sut_val, args.test_step_size) + _rewrap(np_val, args.test_step_size)
    test_datasets = _rewrap(suturing_data.test_dataset, args.test_step_size) + _rewrap(
        needle_data.test_dataset, args.test_step_size
    )
    use_validation = len(val_datasets) > 0

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_kin_features)
    val_loader = (
        DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_kin_features)
        if use_validation
        else None
    )
    test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_kin_features)

    prompts = PROMPT_SETS[args.prompt_type]
    d_text = 512
    model = GVRModulePredKinFeatures(
        gesture_prompts=prompts,
        d_text=d_text,
        d_model=feature_dim,
        num_heads=args.num_heads,
        segment_length=args.segment_length,
        position=args.position,
        dropout=args.dropout,
        use_pooled_branch=args.use_pooled_branch,
        fusion=args.fusion,
        pooled_mode=args.pooled_mode,
        attn_temperature=args.attn_temperature,
        kin_fusion=args.kin_fusion,
        kin_dim_ratio=args.kin_dim_ratio,
    ).to(device)

    pos_weight = None
    if args.use_pos_weight:
        pos_weight, _ = compute_pos_weight_kin_features(
            train_loader, device, window_label_rule=args.window_label_rule
        )
    criterion = build_criterion(
        loss=args.loss,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None

    fold_log = {
        "fold": [],
        "epoch": [],
        "feature_source": [],
        "feature_model": [],
        "feature_root": [],
        "prompt_type": [],
        "split_scheme": [],
        "learning_rate": [],
        "weight_decay": [],
        "train_f1": [],
        "train_accuracy": [],
        "train_jaccard": [],
        "val_f1": [],
        "val_accuracy": [],
        "val_jaccard": [],
        "test_f1": [],
        "test_accuracy": [],
        "test_jaccard": [],
    }

    print(f"\n{'='*60}")
    print(f"Training on Fold {fold} [{args.split_scheme}] ({dataset_variant_suturing} + {dataset_variant_needle})")
    print(f"{'='*60}")
    print(f"[INFO] Using feature source {feature_tag} (model={args.model}, dim={feature_dim}) + kinematics")
    selection_split = "val" if use_validation else "test"
    print(
        f"[INFO] Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset) if use_validation else 0}, "
        f"Test samples: {len(test_loader.dataset)} (model selection on {selection_split} {args.selection_metric})"
    )

    best_select_f1 = 0.0
    best_epoch = None
    best_model_state = None
    best_test_metrics = None
    epochs_without_improvement = 0

    for epoch in range(args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.num_epochs} (LR: {current_lr:.6f})")

        train_f1, train_acc, train_jac = train_kin_features(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=args.max_grad_norm,
            window_label_rule=args.window_label_rule,
        )
        if use_validation:
            val_f1, val_acc, val_jac = test_kin_features(
                model,
                val_loader,
                device,
                debug=False,
                criterion=criterion,
                window_label_rule=args.window_label_rule,
            )
        else:
            val_f1, val_acc, val_jac = float("nan"), float("nan"), float("nan")
        test_f1, test_acc, test_jac = test_kin_features(
            model,
            test_loader,
            device,
            debug=False,
            criterion=criterion,
            window_label_rule=args.window_label_rule,
        )

        fold_log["fold"].append(fold)
        fold_log["epoch"].append(epoch + 1)
        fold_log["feature_source"].append(feature_tag)
        fold_log["feature_model"].append(args.model)
        fold_log["feature_root"].append(feature_source_desc)
        fold_log["prompt_type"].append(args.prompt_type)
        fold_log["split_scheme"].append(args.split_scheme)
        fold_log["learning_rate"].append(current_lr)
        fold_log["weight_decay"].append(args.weight_decay)
        fold_log["train_f1"].append(train_f1)
        fold_log["train_accuracy"].append(train_acc)
        fold_log["train_jaccard"].append(train_jac)
        fold_log["val_f1"].append(val_f1)
        fold_log["val_accuracy"].append(val_acc)
        fold_log["val_jaccard"].append(val_jac)
        fold_log["test_f1"].append(test_f1)
        fold_log["test_accuracy"].append(test_acc)
        fold_log["test_jaccard"].append(test_jac)

        print(f"Train - F1: {train_f1:.4f}, Bal acc: {train_acc:.4f}, Jaccard: {train_jac:.4f}")
        if use_validation:
            print(f"Val   - F1: {val_f1:.4f}, Bal acc: {val_acc:.4f}, Jaccard: {val_jac:.4f}")
        print(f"Test  - F1: {test_f1:.4f}, Bal acc: {test_acc:.4f}, Jaccard: {test_jac:.4f}")

        if use_validation:
            select_score = _selection_score(val_f1, val_acc, args.selection_metric)
        else:
            select_score = _selection_score(test_f1, test_acc, args.selection_metric)
        if select_score > best_select_f1:
            best_select_f1 = select_score
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_test_metrics = (test_f1, test_acc, test_jac)
            epochs_without_improvement = 0
            print(f"  *** New best {selection_split.upper()} {args.selection_metric}: {best_select_f1:.4f} ***")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(f"\n[INFO] Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"\n[INFO] Restored best model from epoch {best_epoch} "
            f"({selection_split} {args.selection_metric}: {best_select_f1:.4f})"
        )
        if best_test_metrics is not None:
            print(
                f"[INFO] Test metrics at selected epoch - F1: {best_test_metrics[0]:.4f}, "
                f"Bal acc: {best_test_metrics[1]:.4f}, Jaccard: {best_test_metrics[2]:.4f}"
            )

    save_dir = CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"gvrkin_{feature_tag}_{args.prompt_type}_{args.split_scheme}_fold_{fold}{tag_suffix}_best.pth",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_model": args.model,
            "feature_source": feature_tag,
            "feature_root": args.feature_root,
            "prompt_type": args.prompt_type,
            "feature_dim": feature_dim,
            "selection_split": selection_split,
            "best_selection_score": best_select_f1,
            "selection_metric": args.selection_metric,
            "best_test_metrics": best_test_metrics,
            "best_epoch": best_epoch,
            "args": vars(args),
        },
        save_path,
    )
    print(f"Best model saved to {save_path}")

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"Fold {fold} training took {elapsed_hours:.2f} hours")

    _fold_df = pd.DataFrame(fold_log)
    for _pre, _f1c, _accc in (("val", "val_f1", "val_accuracy"), ("test", "test_f1", "test_accuracy")):
        _fold_df[f"{_pre}_{args.selection_metric}"] = [
            _selection_score(a, b, args.selection_metric) for a, b in zip(_fold_df[_f1c], _fold_df[_accc])
        ]
    all_fold_logs.append(_fold_df)


final_df = pd.concat(all_fold_logs, ignore_index=True)
final_df.to_csv(csv_path, index=False)
print(f"\n[INFO] Training complete! Logs saved to {csv_path}")

select_col = ('val' if args.val_ratio > 0 else 'test') + '_' + args.selection_metric
print("\n" + "=" * 60)
print(f"SUMMARY - selected by best {select_col} per fold:")
print("=" * 60)
selected_test_f1s = []
for fold_df in all_fold_logs:
    fold_num = int(fold_df["fold"].iloc[0])
    sel_idx = fold_df[select_col].idxmax()
    sel_epoch = int(fold_df.loc[sel_idx, "epoch"])
    sel_test_f1 = float(fold_df.loc[sel_idx, "test_f1"])
    selected_test_f1s.append(sel_test_f1)
    print(
        f"  Fold {fold_num}: best {select_col} = {float(fold_df[select_col].max()):.4f} "
        f"(epoch {sel_epoch}) -> test F1 = {sel_test_f1:.4f}"
    )

print(f"\n  Average Test F1 at selected epochs: {np.mean(selected_test_f1s):.4f} "
      f"(std {np.std(selected_test_f1s):.4f})")


