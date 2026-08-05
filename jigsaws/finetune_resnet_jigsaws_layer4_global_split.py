#!/usr/bin/env python3
"""
Finetune torchvision ResNet on JIGSAWS (Suturing + Needle Passing) segment labels.

Compared to `finetune_gesture_prompt_model.py`:
- All videos are pooled; a single random 80/20 train/test split is done at the video level
  (no LOSO/LOUO training loop).
- Only the last ResNet stage (`backbone.layer4`) is trainable, plus a single linear classifier
  on the backbone embedding (e.g. 2048-d for ResNet50). **No** 128-d projection layer.
- After training one model on the training split, the **same** `state_dict` is written once
  per fold (names match the fold-wise script) so `extract_gesture_prompt_features.py` can
  still resolve `*_fold{k}_best.pth` paths. Extracted frame features are the fine-tuned
  backbone dimension (e.g. 2048), not 128.

Use a dedicated `--save_dir` so these checkpoints do not overwrite per-fold LOSO/LOUO runs.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from torch.cuda.amp import GradScaler

from finetune_gesture_prompt_model import (
    _build_segment_dataset,
    _class_weights_for_label_type,
    build_transforms,
    eval_one_epoch,
    train_one_epoch,
)
from jigsaws_splits import TASKS, get_split_fold_ids, resolve_split_root
from prompt_finetune_datasets import collate_gesture_segments
from prompt_finetune_models import GestureFinetuneConfig, build_gesture_finetune_model, resolve_num_classes

RESNET_BACKBONES = ("resnet18", "resnet50", "resnet101")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _discover_all_video_csvs(data_root: str, tasks: Sequence[str] = TASKS) -> List[str]:
    names: List[str] = []
    for task in tasks:
        err_dir = os.path.join(data_root, task, "errors")
        if not os.path.isdir(err_dir):
            raise FileNotFoundError(f"Missing errors directory: {err_dir}")
        for fn in sorted(os.listdir(err_dir)):
            if fn.endswith(".csv"):
                names.append(fn)
    if not names:
        raise RuntimeError(f"No error CSVs found under {data_root}/{{task}}/errors/")
    return names


def _train_test_split_videos(
    videos: List[str],
    train_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    rng = random.Random(seed)
    vids = list(videos)
    rng.shuffle(vids)
    n_train = int(round(len(vids) * train_ratio))
    n_train = max(1, min(n_train, len(vids) - 1)) if len(vids) >= 2 else 1
    if len(vids) == 1:
        return vids, vids
    return vids[:n_train], vids[n_train:]


def _apply_train_layer4_plus_head(model: torch.nn.Module, meta: Dict) -> None:
    if meta.get("type") != "resnet":
        raise ValueError("This script only supports ResNet backbones (got meta['type'] != 'resnet').")
    backbone = model.backbone
    for p in backbone.parameters():
        p.requires_grad = False
    if not hasattr(backbone, "layer4"):
        raise AttributeError("Backbone has no layer4 (expected torchvision ResNet).")
    for p in backbone.layer4.parameters():
        p.requires_grad = True
    proj = getattr(model, "feature_proj", None)
    if proj is not None:
        for p in proj.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    meta["freeze_backbone_except_layer4"] = True
    meta["trainable_backbone_stages"] = ["layer4"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune ResNet layer4 + linear classifier (no 128-d projection) on all JIGSAWS "
        "videos with global 80/20 split; save identical checkpoints per fold."
    )
    parser.add_argument("--backbone", type=str, default="resnet50", choices=RESNET_BACKBONES)
    parser.add_argument("--label_type", type=str, default="gesture", choices=["gesture", "error"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction of videos for training")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed. Unset leaves RNGs unseeded.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--segment_length", type=int, default=20)
    parser.add_argument("--step_size", type=int, default=12)
    parser.add_argument("--sample_frames", type=int, default=10)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./gesture_prompt_ckpts_layer4_global")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_class_weights", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument(
        "--split_scheme",
        type=str,
        default="loso",
        choices=["loso", "louo"],
        help="Used only for output checkpoint naming (fold ids) and split_root in metadata.",
    )
    parser.add_argument("--split_root", type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
    device = torch.device(args.device)
    args.amp = bool(args.amp == 1)
    os.makedirs(args.save_dir, exist_ok=True)

    num_classes = resolve_num_classes(args.label_type)
    split_root = resolve_split_root(args.split_scheme, split_root=args.split_root, repo_root=".")

    all_videos = _discover_all_video_csvs(args.data_root, tasks=TASKS)
    train_videos, test_videos = _train_test_split_videos(all_videos, args.train_ratio, args.seed)
    print(f"[INFO] Videos total={len(all_videos)} train={len(train_videos)} test={len(test_videos)} "
          f"(train_ratio={args.train_ratio}, seed={args.seed})")
    print(f"[INFO] label_type={args.label_type} num_classes={num_classes} backbone={args.backbone}")
    print(f"[INFO] split_scheme={args.split_scheme} (for checkpoint names only) split_root={split_root}")

    cfg = GestureFinetuneConfig(
        backbone=args.backbone,
        label_type=args.label_type,
        num_classes=num_classes,
        resnet_projected_dim=None,
        resnet_head_dropout=0.0,
    )

    transform = build_transforms(args.backbone)
    train_sets = [_build_segment_dataset(v, args, transform) for v in train_videos]
    test_sets = [_build_segment_dataset(v, args, transform) for v in test_videos]

    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_gesture_segments,
    )
    test_loader = DataLoader(
        ConcatDataset(test_sets),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_gesture_segments,
    )

    if args.use_class_weights == 1:
        class_w = _class_weights_for_label_type(args.label_type, train_sets, device)
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    model, meta = build_gesture_finetune_model(cfg)
    _apply_train_layer4_plus_head(model, meta)
    meta["training_mode"] = "global_video_split"
    meta["train_ratio"] = float(args.train_ratio)
    meta["split_scheme_for_naming"] = args.split_scheme
    meta["replicated_fold_checkpoints_identical"] = True
    meta["seed"] = -1 if args.seed is None else int(args.seed)

    model = model.to(device)
    train_one_epoch._scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Trainable parameters: {n_trainable}")

    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    best_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.num_epochs} (LR={lr:.2e})")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, clip_processor=None)
        test_acc = eval_one_epoch(model, test_loader, device, clip_processor=None)
        print(f"  Train loss={train_loss:.4f} acc={train_acc:.4f} | Test acc={test_acc:.4f}")
        scheduler.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  *** New best test acc={best_acc:.4f} ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("  Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    fold_ids = get_split_fold_ids(args.split_scheme)
    prefix = "gesture_finetune" if args.label_type == "gesture" else "error_finetune"
    args_dict = vars(args).copy()
    args_dict["split_root_resolved"] = split_root

    for fold in fold_ids:
        ckpt_name = f"{prefix}_{args.backbone}_{args.split_scheme}_fold{fold}_best.pth"
        save_path = os.path.join(args.save_dir, ckpt_name)
        meta_out = dict(meta)
        meta_out["fold"] = int(fold)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "meta": meta_out,
                "args": args_dict,
                "best_test_acc": float(best_acc),
                "fold": int(fold),
            },
            save_path,
        )
        print(f"[INFO] Saved {save_path} (replica fold={fold}, best_test_acc={best_acc:.4f})")

    elapsed_h = (time.time() - start) / 3600.0
    print(f"\n[INFO] Done. wall_hours={elapsed_h:.3f} save_dir={args.save_dir}")


if __name__ == "__main__":
    main()
