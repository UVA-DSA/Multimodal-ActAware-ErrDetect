#!/usr/bin/env python3
"""
Finetune a frozen ResNet50 initialized from ImageNet on SAR-RARP50 error labels.

The resulting checkpoint can be passed to `extract_features_sar_rarp50.py --finetuned_ckpt ...`
to extract per-frame projected features from the finetuned model and save them under
`data/SAR_RARP50/vid_features/<model_name>/...`.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from sar_rarp50_error_finetune_utils import (
    DEFAULT_SAR_RARP50_ERROR_FEATURE_DIM,
    SAR_RARP50ErrorFrameDataset,
    build_sar_rarp50_error_model,
    compute_balanced_class_weights,
    get_resnet50_error_transforms,
    list_video_ids_from_pkl_dir,
    split_video_ids,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": (device.type == "cuda"),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _metrics_from_predictions(labels, preds) -> Dict[str, float]:
    if len(labels) == 0:
        return {"f1": 0.0, "acc": 0.0}
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "acc": float(balanced_accuracy_score(labels, preds)),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, scaler: GradScaler):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    use_amp = bool(scaler.is_enabled())

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            _, logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        preds = torch.argmax(logits.detach(), dim=-1).cpu().numpy().astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    metrics = _metrics_from_predictions(all_labels, all_preds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for images, labels in tqdm(loader, desc="Eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            _, logits = model(images)
            loss = criterion(logits, labels)

        total_loss += float(loss.item())
        num_batches += 1
        preds = torch.argmax(logits, dim=-1).cpu().numpy().astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    metrics = _metrics_from_predictions(all_labels, all_preds)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Finetune ResNet50 on SAR-RARP50 error labels")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--use_test_as_val", type=int, default=1, help="If 1, skip val split and use test set for validation")
    parser.add_argument(
        "--train_repeat_factor",
        type=int,
        default=3,
        help="Repeat each training frame this many times per epoch; stochastic train transforms make repeated views different",
    )
    parser.add_argument("--use_class_weights", type=int, default=0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed. Unset leaves RNGs unseeded.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./sar_rarp50_error_ckpts")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name; defaults to a hyperparameter-based name")
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
    device = torch.device(args.device)
    args.use_test_as_val = bool(args.use_test_as_val == 1)
    args.use_class_weights = bool(args.use_class_weights == 1)
    args.amp = bool(args.amp == 1)

    train_split_set = "training_set"
    test_split_set = "testing_set"
    train_pkl_dir = os.path.join(args.data_root, "train_emb_DINOv2")
    test_pkl_dir = os.path.join(args.data_root, "test_emb_DINOv2")
    os.makedirs(args.save_dir, exist_ok=True)

    all_train_video_ids = list_video_ids_from_pkl_dir(train_pkl_dir)
    if args.use_test_as_val:
        train_video_ids = all_train_video_ids
        val_video_ids = None
    else:
        train_video_ids, val_video_ids = split_video_ids(all_train_video_ids, val_ratio=args.val_ratio, seed=args.seed)

    train_ds = SAR_RARP50ErrorFrameDataset(
        data_root=args.data_root,
        split_set=train_split_set,
        pkl_dir=train_pkl_dir,
        video_ids=train_video_ids,
        transform=get_resnet50_error_transforms(train=True),
        repeat_factor=args.train_repeat_factor,
    )
    if args.use_test_as_val:
        val_ds = SAR_RARP50ErrorFrameDataset(
            data_root=args.data_root,
            split_set=test_split_set,
            pkl_dir=test_pkl_dir,
            video_ids=None,
            transform=get_resnet50_error_transforms(train=False),
        )
    else:
        val_ds = SAR_RARP50ErrorFrameDataset(
            data_root=args.data_root,
            split_set=train_split_set,
            pkl_dir=train_pkl_dir,
            video_ids=val_video_ids,
            transform=get_resnet50_error_transforms(train=False),
        )
    test_ds = SAR_RARP50ErrorFrameDataset(
        data_root=args.data_root,
        split_set=test_split_set,
        pkl_dir=test_pkl_dir,
        video_ids=None,
        transform=get_resnet50_error_transforms(train=False),
    )

    train_loader = _build_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = _build_loader(val_ds, args.batch_size, False, args.num_workers, device)
    test_loader = _build_loader(test_ds, args.batch_size, False, args.num_workers, device)

    print(f"[INFO] device={device}")
    print(
        f"[INFO] train_videos={len(train_video_ids)} "
        f"base_train_samples={getattr(train_ds, 'base_num_samples', len(train_ds))} "
        f"effective_train_samples={len(train_ds)} "
        f"train_repeat_factor={args.train_repeat_factor}"
    )
    print(f"[INFO] val_samples={len(val_ds)} test_samples={len(test_ds)}")

    model = build_sar_rarp50_error_model(
        pretrained=True,
        num_classes=2,
        projected_dim=DEFAULT_SAR_RARP50_ERROR_FEATURE_DIM,
        freeze_backbone=True,
    ).to(device)
    print(
        f"[INFO] frozen ResNet50 backbone with projected feature dim={model.hidden_dim} "
        f"(backbone_output_dim={model.backbone_output_dim})"
    )
    if args.use_class_weights:
        class_weights = compute_balanced_class_weights(train_ds.targets, num_classes=2, device=device)
        print(f"[INFO] class_weights={class_weights.detach().cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    run_tag = args.run_name or (
        f"sar_rarp50_resnet50_error_ft_lr{args.learning_rate}_wd{args.weight_decay}_"
        f"bs{args.batch_size}_seed{args.seed}"
    )
    ckpt_path = os.path.join(args.save_dir, f"{run_tag}_best.pth")
    csv_path = os.path.join(args.save_dir, f"{run_tag}_metrics.csv")

    best_val_f1 = 0.0
    best_state = None
    best_epoch = None
    epochs_without_improve = 0
    logs = []
    start_time = time.time()

    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.num_epochs} (LR={lr:.2e})")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = eval_one_epoch(model, val_loader, criterion, device, use_amp=(args.amp and device.type == "cuda"))
        test_metrics = eval_one_epoch(model, test_loader, criterion, device, use_amp=(args.amp and device.type == "cuda"))
        scheduler.step(val_metrics["f1"])

        print(
            f"  Train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} bacc={train_metrics['acc']:.4f}"
        )
        print(f"  Val   loss={val_metrics['loss']:.4f} f1={val_metrics['f1']:.4f} bacc={val_metrics['acc']:.4f}")
        print(f"  Test  loss={test_metrics['loss']:.4f} f1={test_metrics['f1']:.4f} bacc={test_metrics['acc']:.4f}")

        logs.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_f1": train_metrics["f1"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["acc"],
                "test_loss": test_metrics["loss"],
                "test_f1": test_metrics["f1"],
                "test_acc": test_metrics["acc"],
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improve = 0
            print(f"  *** New best val_f1={best_val_f1:.4f} (epoch {best_epoch}) ***")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print("  Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "backbone": "resnet50",
                "label_type": "error",
                "num_classes": 2,
                "feature_dim": model.hidden_dim,
                "projected_dim": model.hidden_dim,
                "backbone_output_dim": model.backbone_output_dim,
                "freeze_backbone": True,
                "dataset": "SAR_RARP50",
            },
            "args": vars(args),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(csv_path, index=False)

    print(f"\n[INFO] Saved checkpoint: {ckpt_path}")
    print(f"[INFO] Saved logs: {csv_path}")
    print(f"[INFO] Total time: {(time.time() - start_time) / 3600.0:.2f} hours")


if __name__ == "__main__":
    main()
