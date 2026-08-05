"""
Training script for GVR Error Detection using pre-extracted features.

This script uses pre-extracted visual features instead of raw images,
significantly speeding up training by avoiding redundant CNN/ViT computation.

Supports features from ResNet and CLIP vision encoders.

Includes:
- Learning rate scheduler (ReduceLROnPlateau)
- Weight decay (L2 regularization)
- Class-weighted loss (pos_weight)
- Early stopping
- Best model checkpointing

Run extract_features.py first to generate the features:
    python extract_features.py --model resnet50
    python extract_features.py --model clip-vit-base-patch32
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import os
import time
import argparse
import copy
import random
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch
from PIL import Image

from model_gvr_features import GVRModulePredFeatures
from dataloader_features import (
    JIGSAWSDataFeatures,
    JIGSAWS_Gesture_Features_SegmentWrapperWithMeta,
    collate_fn_features,
    collate_fn_features_with_meta,
    infer_feature_dim_from_root,
    MODEL_FEATURE_DIMS,
)
from jigsaws_splits import get_split_fold_ids, make_dataset_variant
from util_features import train_features, test_features, compute_pos_weight, build_criterion
from textprompts import *

# Available feature extractor models
AVAILABLE_MODELS = list(MODEL_FEATURE_DIMS.keys())

# Available prompt sets (from textprompts.py)
PROMPT_SETS = {
    "error": error_prompt,
    "gesture_error": gesture_error_prompt,
    "gesture": gesture_prompt,
    "lowlevel_gesture_error": lowlevel_gesture_error_prompt,
    "context": Context_Prompt,
}
AVAILABLE_PROMPT_TYPES = list(PROMPT_SETS.keys())

parser = argparse.ArgumentParser(
    description="Train GVRModulePredFeatures model with pre-extracted features.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Train with ResNet50 features (default)
  python train_eval_gvr_error_features.py --learning_rate 0.001 --model resnet50
  
  # Train with CLIP ViT-B/32 features
  python train_eval_gvr_error_features.py --learning_rate 0.001 --model clip-vit-base-patch32
  
  # Train with CLIP ViT-L/14 features
  python train_eval_gvr_error_features.py --learning_rate 0.001 --model clip-vit-large-patch14
"""
)
parser.add_argument("--position", type=int, default=1, help="Use positional encoding if 1")
parser.add_argument("--learning_rate", type=float, default=0.0001293, help="Learning rate for tuning")
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
parser.add_argument("--weight_decay", type=float, default=0.01549407, help="Weight decay (L2 regularization)")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
parser.add_argument("--use_pos_weight", type=int, default=1, help="Use class-weighted loss if 1")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
parser.add_argument("--debug_eval", type=int, default=0, help="If 1, print one eval batch's feature/mask/logit stats per epoch")
parser.add_argument("--use_pooled_branch", type=int, default=1, help="Use pooled features in final prediction if 1")
parser.add_argument("--segment_length", type=int, default=20, help="Window length in samples for sliding segments")
parser.add_argument("--train_step_size", type=int, default=10, help="Step size for training sliding windows")
parser.add_argument("--test_step_size", type=int, default=10, help="Step size for test sliding windows")
parser.add_argument(
    "--frame_subsample",
    type=int,
    default=2,
    help="Keep every k-th stored frame. Stored JIGSAWS frames are 10 Hz; the default 2 gives the paper's 5 Hz rate.",
)
parser.add_argument("--seed", type=int, default=None,
                    help="Optional random seed. Unset (default) leaves all RNGs unseeded.")
parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.2,
    help="Fraction of training videos (per task) held out as a validation set. "
         "When > 0, early stopping and model selection use validation F1 and the test set is only "
         "evaluated for reporting. When 0, the legacy behavior (selection on test F1) is kept.",
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
parser.add_argument("--visualize_test", type=int, default=0, help="If 1, save test attention/labels/frame visualizations")
parser.add_argument("--vis_ratio", type=float, default=0.3, help="Fraction of test windows to visualize")
parser.add_argument("--vis_outdir", type=str, default="./outputs/jigsaws/vis_test_attn", help="Output directory for visualizations")
parser.add_argument("--vis_seed", type=int, default=None, help="Optional random seed for visualization sampling")
parser.add_argument("--vis_max_windows", type=int, default=0, help="Max windows to visualize per fold (0 = no cap)")
parser.add_argument(
    "--prompt_encoder",
    type=str,
    default="clip",
    choices=["clip", "surgvlp"],
    help="Prompt encoder to use for gesture prompts (clip or surgvlp)",
)
parser.add_argument(
    "--prompt_emb_path",
    type=str,
    default="./data/prompt_features_surgvlp/prompts.pt",
    help="Path to precomputed prompt embeddings (used when --prompt_encoder surgvlp)",
)
parser.add_argument(
    "--window_label_rule",
    type=str,
    default="any_error",
    choices=["majority", "any_error"],
    help="How to derive one label per window from per-frame labels: majority (tie->error) or any_error (1 if any valid frame is error).",
)
# --- architectural variants (see model_gvr_features.py) ---
parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "gated", "film"],
                    help="How the pooled video summary is fused with the refined prompt features")
parser.add_argument("--pooled_mode", type=str, default="mean", choices=["tcn", "mean", "attn"],
                    help="Pooling used for the direct video summary branch")
parser.add_argument("--attn_temperature", type=float, default=0.7,
                    help="Temperature for the prompt-to-frame attention (values < 1 sharpen attention)")
# --- loss variants ---
parser.add_argument("--loss", type=str, default="focal", choices=["wbce", "focal"],
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
args.visualize_test = True if args.visualize_test == 1 else False
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

# Settings
num_epochs = args.num_epochs
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
print(f"[INFO] Max grad norm: {args.max_grad_norm}")
print(f"[INFO] Early stopping patience: {args.patience}")
print(f"[INFO] Use pos_weight: {args.use_pos_weight}")
print(f"[INFO] Use pooled branch: {args.use_pooled_branch}")
print(f"[INFO] Segment length: {args.segment_length}")
print(f"[INFO] Train step size: {args.train_step_size}")
print(f"[INFO] Test step size: {args.test_step_size}")
print(f"[INFO] Seed: {args.seed}")
print(f"[INFO] Split scheme: {args.split_scheme}")
print(f"[INFO] Split root: {args.split_root or '(auto)'}")
print(f"[INFO] Window label rule: {args.window_label_rule}")
print(f"[INFO] Visualize test: {args.visualize_test}")
print(f"[INFO] Prompt encoder: {args.prompt_encoder}")
if args.visualize_test:
    print(f"[INFO] Visualize ratio: {args.vis_ratio}")
    print(f"[INFO] Visualize outdir: {args.vis_outdir}")
    print(f"[INFO] Visualize seed: {args.vis_seed}")
    print(f"[INFO] Visualize max windows: {args.vis_max_windows}")


def _gesture_id_map():
    return {i + 1: name for i, name in enumerate(gesture_prompt)}


def _align_mask_to_len(mask: torch.Tensor, length: int) -> torch.Tensor:
    if mask.size(0) == length:
        return mask
    if mask.size(0) > length:
        return mask[:length]
    pad = torch.zeros(length - mask.size(0), device=mask.device, dtype=mask.dtype)
    return torch.cat([mask, pad], dim=0)


def _load_frame(task: str, video_name: str, frame_num: int) -> Image.Image | None:
    frame_path = os.path.join("./data/vid_frames", task, video_name, f"frame_{frame_num}.png")
    if not os.path.exists(frame_path):
        return None
    return Image.open(frame_path).convert("RGB")


def _frame_tick_config(frame_nums: list[int], max_ticks: int = 8) -> tuple[list[int], list[str]]:
    if not frame_nums:
        return [], []
    step = max(1, int(np.ceil(len(frame_nums) / max_ticks)))
    tick_idx = list(range(0, len(frame_nums), step))
    if tick_idx[-1] != len(frame_nums) - 1:
        tick_idx.append(len(frame_nums) - 1)
    tick_labels = [str(int(frame_nums[i])) for i in tick_idx]
    return tick_idx, tick_labels


def _save_attention_figure(
    out_path: str,
    attn: np.ndarray,
    frame_nums: list[int],
    gestures: list[int],
    errors: list[int],
    prob: float,
):
    if attn.size == 0 or not frame_nums:
        return

    tick_idx, tick_labels = _frame_tick_config(frame_nums)
    gesture_palette = [
        "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
        "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9",
        "#0072B2", "#009E73", "#E69F00", "#C44E52", "#7A68A6",
    ]
    font_sizes = {
        "label": 16,
        "ticks": 13,
        "legend": 12,
        "annot": 13,
    }

    fig_width = max(11.5, len(frame_nums) * 0.42)
    fig = plt.figure(figsize=(fig_width, 6.8), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[40, 1.2],
        height_ratios=[6.2, 0.45, 0.45],
        wspace=0.06,
        hspace=0.08,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.24, top=0.94)

    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(grid[2, 0], sharex=ax0)
    cax = fig.add_subplot(grid[0, 1])
    axes = [ax0, ax1, ax2]

    positive_attn = attn[attn > 0]
    attn_vmax = float(attn.max()) if float(attn.max()) > 0 else 1.0
    attn_vmin = float(np.quantile(positive_attn, 0.2)) if positive_attn.size else 0.0
    if attn_vmin >= attn_vmax:
        attn_vmin = 0.0
    attn_norm = PowerNorm(gamma=1.8, vmin=attn_vmin, vmax=attn_vmax, clip=True)
    im = ax0.imshow(
        attn,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=attn_norm,
    )
    ax0.set_facecolor("#FBFBFD")
    ax0.set_ylabel("Prompt", fontsize=font_sizes["label"], fontweight="semibold")
    ax0.set_yticks(np.arange(attn.shape[0]))
    ax0.set_yticklabels([str(i + 1) for i in range(attn.shape[0])], fontsize=font_sizes["ticks"])
    ax0.tick_params(axis="x", labelbottom=False, bottom=False)
    ax0.tick_params(axis="y", labelsize=font_sizes["ticks"], width=0.9, length=4, direction="out")
    ax0.text(
        1.0,
        1.03,
        f"Pred. prob: {prob:.2f}",
        transform=ax0.transAxes,
        ha="right",
        va="bottom",
        fontsize=font_sizes["annot"],
        fontweight="semibold",
        color="#2D3142",
    )
    for spine in ax0.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#5C5C5C")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention Weight", fontsize=font_sizes["label"] - 1, fontweight="semibold")
    cbar.ax.tick_params(labelsize=font_sizes["ticks"], width=0.9, length=4, direction="out")
    cbar.outline.set_linewidth(0.8)

    unique_gestures = sorted(set(gestures))
    gesture_colors = {gid: gesture_palette[i % len(gesture_palette)] for i, gid in enumerate(unique_gestures)}
    gesture_cmap = ListedColormap([gesture_colors[g] for g in unique_gestures])
    gesture_to_idx = {g: i for i, g in enumerate(unique_gestures)}
    gesture_idx = np.array([gesture_to_idx[g] for g in gestures], dtype=int)
    ax1.imshow(
        np.array([gesture_idx]),
        aspect="auto",
        interpolation="nearest",
        cmap=gesture_cmap,
        vmin=0,
        vmax=max(len(unique_gestures) - 1, 1),
    )
    ax1.set_yticks([])
    ax1.set_facecolor("white")
    ax1.set_ylabel(
        "G",
        fontsize=font_sizes["label"],
        fontweight="semibold",
        rotation=270,
        labelpad=18,
    )
    ax1.yaxis.set_label_position("right")
    ax1.tick_params(axis="x", labelbottom=False, bottom=False)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    error_cmap = ListedColormap(["#F1F1F1", "#C23B3B"])
    ax2.imshow(
        np.array([errors]),
        aspect="auto",
        interpolation="nearest",
        cmap=error_cmap,
        vmin=0,
        vmax=1,
    )
    ax2.set_yticks([])
    ax2.set_facecolor("white")
    ax2.set_ylabel("E", fontsize=font_sizes["label"], fontweight="semibold")
    ax2.set_xlabel("Frame Number", fontsize=font_sizes["label"], fontweight="semibold")
    ax2.tick_params(axis="x", labelsize=font_sizes["ticks"], width=0.9, length=5, direction="out")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    for ax in axes:
        ax.set_xlim(-0.5, len(frame_nums) - 0.5)
        ax.set_xticks(tick_idx)

    ax2.set_xticklabels(tick_labels)

    legend_patches = [
        Patch(
            color=gesture_colors[gid],
            label=f"G{gid}",
        )
        for gid in unique_gestures
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=min(6, len(legend_patches)),
            fontsize=font_sizes["legend"],
            title="Gestures",
            title_fontsize=font_sizes["legend"] + 1,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.5,
            columnspacing=1.1,
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_frames_grid(out_path: str, frames: list[tuple[int, Image.Image]], cols: int = 6):
    if not frames:
        return
    cols = min(cols, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.35), squeeze=False)
    fig.patch.set_facecolor("white")
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx < len(frames):
            frame_num, frame_img = frames[idx]
            ax.imshow(frame_img)
            ax.set_title(f"{int(frame_num)}", fontsize=13, pad=6, fontweight="semibold")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_color("#D8D8D8")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@torch.no_grad()
def visualize_test_windows(model, test_datasets, fold_id: int):
    os.makedirs(args.vis_outdir, exist_ok=True)
    out_dir = os.path.join(args.vis_outdir, f"{args.split_scheme}_fold_{fold_id}")
    os.makedirs(out_dir, exist_ok=True)
    if args.vis_seed is not None:
        random.seed(args.vis_seed + fold_id)
        np.random.seed(args.vis_seed + fold_id)

    vis_sets = [
        JIGSAWS_Gesture_Features_SegmentWrapperWithMeta(
            ds.original_dataset, segment_length=args.segment_length, step_size=args.test_step_size
        )
        for ds in test_datasets
    ]
    vis_loader = DataLoader(
        ConcatDataset(vis_sets),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_features_with_meta,
    )

    model.eval()
    saved = 0
    for batch_idx, batch in enumerate(vis_loader):
        if batch is None:
            continue
        features, masks, labels, gestures, frame_nums, video_names, tasks = batch
        features = features.to(device)
        masks = masks.to(device)
        outputs = model(features, masks=masks, return_attn=True)
        logits, attn_weights = outputs
        probs = torch.sigmoid(logits).detach().cpu()
        attn_weights = attn_weights.detach().cpu()
        masks = masks.detach().cpu()
        labels = labels.detach().cpu()
        gestures = gestures.detach().cpu()
        frame_nums = frame_nums.detach().cpu()

        for i in range(features.size(0)):
            if random.random() > args.vis_ratio:
                continue
            valid_len = int(masks[i].sum().item())
            if valid_len <= 0:
                continue
            if args.vis_max_windows and saved >= args.vis_max_windows:
                return

            even_idx = np.arange(0, valid_len, 2, dtype=int)
            attn = attn_weights[i, :, :valid_len].numpy()[:, even_idx]
            gesture_seq_full = gestures[i, :valid_len].tolist()
            error_seq_full = labels[i, :valid_len].tolist()
            frame_seq_full = frame_nums[i, :valid_len].tolist()
            gesture_seq = [gesture_seq_full[idx] for idx in even_idx]
            error_seq = [error_seq_full[idx] for idx in even_idx]
            frame_seq = [frame_seq_full[idx] for idx in even_idx]
            video_name = video_names[i]
            task = tasks[i]

            base_name = f"{video_name}_b{batch_idx}_i{i}_len{valid_len}"
            attn_path = os.path.join(out_dir, f"{base_name}_attn.png")
            frames_path = os.path.join(out_dir, f"{base_name}_frames.png")

            prob = float(probs[i].item())
            _save_attention_figure(attn_path, attn, frame_seq, gesture_seq, error_seq, prob)

            frames = []
            for fn in frame_seq:
                img = _load_frame(task, video_name, fn)
                if img is not None:
                    frames.append((fn, img))
            _save_frames_grid(frames_path, frames)

            saved += 1

# CSV and output file paths
LOG_DIR = "./outputs/jigsaws/logs"
CKPT_DIR = "./outputs/jigsaws/checkpoints"
tag_suffix = f"_{args.log_tag}" if args.log_tag else ""
csv_filename = (
    f"training_gvrerr_{feature_tag}_{args.prompt_type}_{args.split_scheme}_"
    f"lr_{args.learning_rate}_wd_{args.weight_decay}{tag_suffix}.csv"
)
csv_path = os.path.join(LOG_DIR, csv_filename)
os.makedirs(LOG_DIR, exist_ok=True)


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
for i in fold_ids:
    start_time = time.time()
    dataset_variant_suturing = make_dataset_variant("Suturing", args.split_scheme, i)
    dataset_variant_needle = make_dataset_variant("Needle_Passing", args.split_scheme, i)

    # Load datasets (using feature-based dataloader with specified model)
    suturing_data = JIGSAWSDataFeatures(
        dataset_variant_suturing,
        model_name=args.model,
        split_root=args.split_root,
        feature_root=args.feature_root,
        frame_subsample=args.frame_subsample,
    )
    needle_data = JIGSAWSDataFeatures(
        dataset_variant_needle,
        model_name=args.model,
        split_root=args.split_root,
        feature_root=args.feature_root,
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

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn_features)
    val_loader = (
        DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size,
                   shuffle=False, collate_fn=collate_fn_features)
        if use_validation
        else None
    )
    test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn_features)

    prompts = PROMPT_SETS[args.prompt_type]
    prompt_features = None
    if args.prompt_encoder == "surgvlp":
        if not os.path.exists(args.prompt_emb_path):
            raise FileNotFoundError(f"Prompt embeddings not found: {args.prompt_emb_path}")
        blob = torch.load(args.prompt_emb_path, map_location="cpu")
        if "prompt_types" not in blob or args.prompt_type not in blob["prompt_types"]:
            raise KeyError(f"Prompt type '{args.prompt_type}' not found in {args.prompt_emb_path}")
        prompt_entry = blob["prompt_types"][args.prompt_type]
        prompt_features = prompt_entry["text_emb"]
        if not torch.is_tensor(prompt_features):
            raise TypeError("Loaded prompt embeddings are not a torch.Tensor")
        loaded_prompts = prompt_entry.get("prompts", [])
        if len(loaded_prompts) != len(prompts):
            print(
                f"[WARN] Prompt count mismatch: file has {len(loaded_prompts)} prompts, "
                f"but textprompts has {len(prompts)}. Ensure consistent ordering."
            )
        d_text = int(prompt_features.size(1))
    else:
        d_text = 512
    
    # Use feature-based model with correct feature dimension and dropout
    model = GVRModulePredFeatures(
        gesture_prompts=prompts,
        d_text=d_text,
        d_model=feature_dim,
        num_heads=args.num_heads,
        segment_length=args.segment_length,
        position=args.position,
        dropout=args.dropout,
        prompt_features=prompt_features,
        use_pooled_branch=args.use_pooled_branch,
        fusion=args.fusion,
        pooled_mode=args.pooled_mode,
        attn_temperature=args.attn_temperature,
    ).to(device)

    # Loss (optionally class-weighted, focal, and/or label-smoothed)
    pos_weight = None
    if args.use_pos_weight:
        pos_weight, _ = compute_pos_weight(
            train_loader, device, window_label_rule=args.window_label_rule
        )
    criterion = build_criterion(
        loss=args.loss,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (disabled; test-only evaluation)
    scheduler = None

    # Epoch-wise logs for this fold
    fold_log = {
        'fold': [],
        'epoch': [],
        'feature_source': [],
        'feature_model': [],
        'feature_root': [],
        'prompt_type': [],
        'split_scheme': [],
        'learning_rate': [],
        'weight_decay': [],
        'train_f1': [],
        'train_accuracy': [],
        'train_jaccard': [],
        'val_f1': [],
        'val_accuracy': [],
        'val_jaccard': [],
        'test_f1': [],
        'test_accuracy': [],
        'test_jaccard': []
    }

    print(f"\n{'='*60}")
    print(f"Training on Fold {i} [{args.split_scheme}] ({dataset_variant_suturing} + {dataset_variant_needle})")
    print(f"{'='*60}")
    print(f"[INFO] Using feature source {feature_tag} (model={args.model}, dim={feature_dim})")
    selection_split = "val" if use_validation else "test"
    print(
        f"[INFO] Train samples: {len(train_loader.dataset)}, "
        f"Val samples: {len(val_loader.dataset) if use_validation else 0}, "
        f"Test samples: {len(test_loader.dataset)} (model selection on {selection_split} {args.selection_metric})"
    )

    # Early stopping and best model tracking
    best_select_f1 = 0.0
    best_epoch = None
    best_model_state = None
    best_test_metrics = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")

        train_f1, train_accuracy, train_jaccard = train_features(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_grad_norm=args.max_grad_norm,
            window_label_rule=args.window_label_rule,
        )
        if use_validation:
            val_f1, val_accuracy, val_jaccard = test_features(
                model,
                val_loader,
                device,
                debug=False,
                criterion=criterion,
                window_label_rule=args.window_label_rule,
            )
        else:
            val_f1, val_accuracy, val_jaccard = float("nan"), float("nan"), float("nan")
        test_f1, test_accuracy, test_jaccard = test_features(
            model,
            test_loader,
            device,
            debug=False,
            criterion=criterion,
            window_label_rule=args.window_label_rule,
        )

        fold_log['fold'].append(i)
        fold_log['epoch'].append(epoch + 1)
        fold_log['feature_source'].append(feature_tag)
        fold_log['feature_model'].append(args.model)
        fold_log['feature_root'].append(feature_source_desc)
        fold_log['prompt_type'].append(args.prompt_type)
        fold_log['split_scheme'].append(args.split_scheme)
        fold_log['learning_rate'].append(current_lr)
        fold_log['weight_decay'].append(args.weight_decay)
        fold_log['train_f1'].append(train_f1)
        fold_log['train_accuracy'].append(train_accuracy)
        fold_log['train_jaccard'].append(train_jaccard)
        fold_log['val_f1'].append(val_f1)
        fold_log['val_accuracy'].append(val_accuracy)
        fold_log['val_jaccard'].append(val_jaccard)
        fold_log['test_f1'].append(test_f1)
        fold_log['test_accuracy'].append(test_accuracy)
        fold_log['test_jaccard'].append(test_jaccard)

        print(f"Train - F1: {train_f1:.4f}, Bal acc: {train_accuracy:.4f}, Jaccard: {train_jaccard:.4f}")
        if use_validation:
            print(f"Val   - F1: {val_f1:.4f}, Bal acc: {val_accuracy:.4f}, Jaccard: {val_jaccard:.4f}")
        print(f"Test  - F1: {test_f1:.4f}, Bal acc: {test_accuracy:.4f}, Jaccard: {test_jaccard:.4f}")

        # Check for improvement on the selection split
        if use_validation:
            select_score = _selection_score(val_f1, val_accuracy, args.selection_metric)
        else:
            select_score = _selection_score(test_f1, test_accuracy, args.selection_metric)
        if select_score > best_select_f1:
            best_select_f1 = select_score
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_test_metrics = (test_f1, test_accuracy, test_jaccard)
            epochs_without_improvement = 0
            print(f"  *** New best {selection_split.upper()} {args.selection_metric}: {best_select_f1:.4f} ***")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\n[INFO] Early stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
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

    if args.visualize_test:
        visualize_test_windows(model, test_datasets, i)

    # Save best model
    save_dir = CKPT_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"gvrerr_{feature_tag}_{args.prompt_type}_{args.split_scheme}_fold_{i}{tag_suffix}_best.pth",
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_model': args.model,
        'feature_source': feature_tag,
        'feature_root': args.feature_root,
        'prompt_type': args.prompt_type,
        'feature_dim': feature_dim,
        'selection_split': selection_split,
        'best_selection_score': best_select_f1,
        'selection_metric': args.selection_metric,
        'best_test_metrics': best_test_metrics,
        'best_epoch': best_epoch,
        'args': vars(args),
    }, save_path)
    print(f"Best model saved to {save_path}")

    # Log fold duration
    elapsed_hours = (time.time() - start_time) / 3600
    print(f"Fold {i} training took {elapsed_hours:.2f} hours")

    # Append DataFrame for this fold
    _fold_df = pd.DataFrame(fold_log)
    for _pre, _f1c, _accc in (("val", "val_f1", "val_accuracy"), ("test", "test_f1", "test_accuracy")):
        _fold_df[f"{_pre}_{args.selection_metric}"] = [
            _selection_score(a, b, args.selection_metric) for a, b in zip(_fold_df[_f1c], _fold_df[_accc])
        ]
    all_fold_logs.append(_fold_df)

# Save all epoch-level logs to CSV
final_df = pd.concat(all_fold_logs, ignore_index=True)
final_df.to_csv(csv_path, index=False)
print(f"\n[INFO] Training complete! Logs saved to {csv_path}")

# Print summary
select_col = ('val' if args.val_ratio > 0 else 'test') + '_' + args.selection_metric
print("\n" + "="*60)
print(f"SUMMARY - selected by best {select_col} per fold:")
print("="*60)
selected_test_f1s = []
for fold_df in all_fold_logs:
    fold_num = fold_df['fold'].iloc[0]
    sel_idx = fold_df[select_col].idxmax()
    sel_epoch = fold_df.loc[sel_idx, 'epoch']
    sel_test_f1 = fold_df.loc[sel_idx, 'test_f1']
    selected_test_f1s.append(sel_test_f1)
    print(
        f"  Fold {fold_num}: best {select_col} = {fold_df[select_col].max():.4f} "
        f"(epoch {sel_epoch}) -> test F1 = {sel_test_f1:.4f}"
    )
print(f"\n  Average Test F1 at selected epochs: {np.mean(selected_test_f1s):.4f} "
      f"(std {np.std(selected_test_f1s):.4f})")
