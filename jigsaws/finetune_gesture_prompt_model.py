#!/usr/bin/env python3
"""
Finetune a vision backbone on JIGSAWS gesture or error labels and save checkpoints for feature extraction.

Supported targets:
- gesture: 8-way gesture classification
- error: binary error / no-error classification

Backbones:
- ResNet (torchvision):
  - frozen backbone; trains projected head + classifier head
- CLIP vision (transformers): trains LoRA adapters on attention projections + classifier head

This produces a checkpoint that can be used by `extract_gesture_prompt_features.py`
to dump per-frame hidden embeddings for all videos.
"""

from __future__ import annotations

import argparse
import os
import time
import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from jigsaws_splits import get_split_fold_ids, load_split_records, make_dataset_variant
from prompt_finetune_datasets import GestureDatasetConfig, GestureSegmentDataset, collate_gesture_segments
from prompt_finetune_models import (
    DEFAULT_ERROR_RESNET_PROJECTED_DIM,
    GestureFinetuneConfig,
    apply_lora_to_clip_vision,
    build_gesture_finetune_model,
    resolve_num_classes,
)


AVAILABLE_BACKBONES = [
    "resnet18",
    "resnet50",
    "resnet101",
    "clip-vit-base-patch32",
    "clip-vit-base-patch16",
    "clip-vit-large-patch14",
    "clip-vit-large-patch14-336",
]


def build_transforms(backbone: str):
    # Keep transforms aligned with existing codebase expectations
    from torchvision import transforms

    if backbone.startswith("clip-vit"):
        # Match CLIP preprocessing (mean/std) so we can pass tensors directly to CLIPVisionModel.
        # See CLIPImageProcessor defaults.
        size = 336 if backbone.endswith("336") else 224
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )

    # ResNet-style normalization
    return transforms.Compose(
        [
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _gesture_class_weights_default(device: torch.device) -> torch.Tensor:
    # Reuse weights from old gesture_pretrain/gesture_pre.py (kept as default).
    return torch.tensor([3.2328, 0.5648, 0.5716, 0.7878, 2.5338, 0.5752, 1.9531, 3.9062], device=device)


def _balanced_class_weights_from_targets(targets: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    if len(targets) == 0:
        raise ValueError("Cannot compute class weights from an empty target list")
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=int(num_classes))
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _class_weights_for_label_type(label_type: str, train_sets: List[GestureSegmentDataset], device: torch.device) -> torch.Tensor:
    if label_type == "gesture":
        return _gesture_class_weights_default(device)

    targets: List[int] = []
    for ds in train_sets:
        targets.extend(int(x) for x in getattr(ds, "segment_targets", []))
    weights = _balanced_class_weights_from_targets(targets, num_classes=resolve_num_classes(label_type), device=device)
    print(f"[INFO] Using balanced class weights for label_type={label_type}: {weights.detach().cpu().tolist()}")
    return weights


def _build_trainable_model(cfg: GestureFinetuneConfig, device: torch.device) -> tuple[torch.nn.Module, Dict, object]:
    model, meta = build_gesture_finetune_model(cfg)
    clip_processor = None

    if meta["type"] == "resnet":
        for p in model.backbone.parameters():
            p.requires_grad = False
        if getattr(model, "feature_proj", None) is None:
            raise RuntimeError("ResNet finetuning now requires a projected feature head")
        for p in model.feature_proj.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
        meta["freeze_backbone"] = True
    else:
        for p in model.vision.parameters():
            p.requires_grad = False
        replaced = apply_lora_to_clip_vision(
            model.vision,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
        for p in model.classifier.parameters():
            p.requires_grad = True
        clip_processor = True
        meta["lora_replaced_modules"] = replaced

    model = model.to(device)
    return model, meta, clip_processor


def _resolve_task_name(video_name: str) -> str:
    return "Suturing" if video_name.startswith("Suturing") else "Needle_Passing"


def _build_segment_dataset(video_name: str, args, transform):
    task = _resolve_task_name(video_name)
    ds_cfg = GestureDatasetConfig(
        data_root=args.data_root,
        task=task,
        segment_length=args.segment_length,
        step_size=args.step_size,
        sample_frames=args.sample_frames,
        label_type=args.label_type,
    )
    clean_name = video_name[:-4] if video_name.endswith(".csv") else video_name
    return GestureSegmentDataset(clean_name, ds_cfg, transform=transform)


def _masked_lse_pool_logits(logits: torch.Tensor, masks: torch.Tensor, tau: float) -> torch.Tensor:
    # MIL-style pooling: smooth approximation of max over valid frames.
    masks_f = masks.unsqueeze(-1)
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~masks_f, neg_inf)
    temp = max(float(tau), 1e-6)
    pooled = torch.logsumexp(masked_logits / temp, dim=1) * temp
    valid_counts = masks.sum(dim=1, keepdim=True).clamp_min(1).to(logits.dtype)
    pooled = pooled - valid_counts.log()
    return pooled


def train_one_epoch(model, loader, criterion, optimizer, device, clip_processor=None):
    model.train()
    total_loss = 0.0
    n = 0
    all_preds = []
    all_labels = []

    scaler = getattr(train_one_epoch, "_scaler", None)
    use_amp = bool(scaler is not None and scaler.is_enabled())

    for batch in tqdm(loader, desc="Training"):
        if batch is None:
            continue
        images, masks, target_idx = batch
        masks = masks.to(device)  # (B,T)
        target_idx = target_idx.to(device)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            # Prepare input depending on backbone
            if clip_processor is not None:
                # We pre-normalize via torchvision transforms; just pass as pixel_values.
                pixel_values = images.to(device, non_blocking=True)
                _, logits = model(pixel_values)  # logits: (B,T,C)
            else:
                images = images.to(device, non_blocking=True)
                _, logits = model(images)  # logits: (B,T,C)

            seg_logits = _masked_lse_pool_logits(logits, masks, tau=0.7)  # (B,C)

            loss = criterion(seg_logits, target_idx)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n += 1
        preds = torch.argmax(seg_logits.detach(), dim=-1).cpu().numpy().astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(target_idx.detach().cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / max(n, 1)
    acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if len(all_labels) else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, device, clip_processor=None):
    model.eval()
    all_preds = []
    all_labels = []

    scaler = getattr(train_one_epoch, "_scaler", None)
    use_amp = bool(scaler is not None and scaler.is_enabled())

    for batch in tqdm(loader, desc="Eval"):
        if batch is None:
            continue
        images, masks, target_idx = batch
        masks = masks.to(device)
        target_idx = target_idx.to(device)

        with autocast(enabled=use_amp):
            if clip_processor is not None:
                pixel_values = images.to(device, non_blocking=True)
                _, logits = model(pixel_values)
            else:
                images = images.to(device, non_blocking=True)
                _, logits = model(images)

        seg_logits = _masked_lse_pool_logits(logits, masks, tau=0.7)

        preds = torch.argmax(seg_logits, dim=-1).cpu().numpy().astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(target_idx.cpu().numpy().astype(int).tolist())

    acc = float((np.array(all_preds) == np.array(all_labels)).mean()) if len(all_labels) else 0.0
    return acc


def main():
    parser = argparse.ArgumentParser(description="Finetune prompt model on JIGSAWS gesture or error labels")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=AVAILABLE_BACKBONES)
    parser.add_argument("--label_type", type=str, default="gesture", choices=["gesture", "error"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--resnet_head_dropout", type=float, default=0.2, help="Dropout in ResNet projected head")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--segment_length", type=int, default=20)
    parser.add_argument("--step_size", type=int, default=12)
    parser.add_argument("--sample_frames", type=int, default=10, help="Uniformly sample this many frames from each window")
    parser.add_argument("--amp", type=int, default=1, help="Use AMP if 1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./gesture_prompt_ckpts")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_class_weights", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--split_scheme", type=str, default="loso", choices=["loso", "louo"])
    parser.add_argument("--split_root", type=str, default=None,
                        help="Optional root containing split CSVs; defaults to ./LOSO or ./LOUO from --split_scheme")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.amp = True if args.amp == 1 else False
    os.makedirs(args.save_dir, exist_ok=True)
    num_classes = resolve_num_classes(args.label_type)
    print(f"[INFO] Split scheme: {args.split_scheme}")
    print(f"[INFO] Split root: {args.split_root or '(auto)'}")
    print(f"[INFO] Label type: {args.label_type} (num_classes={num_classes})")

    cfg = GestureFinetuneConfig(
        backbone=args.backbone,
        label_type=args.label_type,
        num_classes=num_classes,
        resnet_projected_dim=(DEFAULT_ERROR_RESNET_PROJECTED_DIM if args.backbone.startswith("resnet") else None),
        resnet_head_dropout=args.resnet_head_dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Transforms/datasets
    transform = build_transforms(args.backbone)

    fold_logs: List[Dict] = []

    for fold in get_split_fold_ids(args.split_scheme):
        start = time.time()
        print(f"\n{'='*60}\nFold {fold}\n{'='*60}")
        dataset_variant_s = make_dataset_variant("Suturing", args.split_scheme, fold)
        dataset_variant_n = make_dataset_variant("Needle_Passing", args.split_scheme, fold)

        split_s = load_split_records(dataset_variant_s, split_root=args.split_root)
        split_n = load_split_records(dataset_variant_n, split_root=args.split_root)
        train_videos = list(split_s["train"]) + list(split_n["train"])
        test_videos = list(split_s["test"]) + list(split_n["test"])

        train_sets = []
        test_sets = []

        for v in train_videos:
            train_sets.append(_build_segment_dataset(v, args, transform))

        for v in test_videos:
            test_sets.append(_build_segment_dataset(v, args, transform))

        train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, collate_fn=collate_gesture_segments)
        test_loader = DataLoader(ConcatDataset(test_sets), batch_size=args.batch_size, shuffle=False, collate_fn=collate_gesture_segments)

        # Loss/optim
        if args.use_class_weights == 1:
            class_w = _class_weights_for_label_type(args.label_type, train_sets, device)
            criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_acc = 0.0
        best_state = None
        epochs_no_improve = 0

        model_fold, meta_fold, clip_processor_fold = _build_trainable_model(cfg, device)
        # Attach a GradScaler to the train function (simple way to avoid threading scaler through many calls)
        train_one_epoch._scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
        trainable_params = [p for p in model_fold.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

        for epoch in range(args.num_epochs):
            lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch+1}/{args.num_epochs} (LR={lr:.2e})")
            train_loss, train_acc = train_one_epoch(model_fold, train_loader, criterion, optimizer, device, clip_processor=clip_processor_fold)
            test_acc = eval_one_epoch(model_fold, test_loader, device, clip_processor=clip_processor_fold)
            print(f"  Train loss={train_loss:.4f} acc={train_acc:.4f} | Test acc={test_acc:.4f}")

            scheduler.step(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_state = copy.deepcopy(model_fold.state_dict())
                epochs_no_improve = 0
                print(f"  *** New best test acc={best_acc:.4f} ***")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print("  Early stopping")
                    break

        # Save best checkpoint
        prefix = "gesture_finetune" if args.label_type == "gesture" else "error_finetune"
        ckpt_name = f"{prefix}_{args.backbone}_{args.split_scheme}_fold{fold}_best.pth"
        save_path = os.path.join(args.save_dir, ckpt_name)
        if best_state is not None:
            model_fold.load_state_dict(best_state)
        torch.save(
            {
                "state_dict": model_fold.state_dict(),
                "meta": meta_fold,
                "args": vars(args),
                "best_test_acc": best_acc,
                "fold": fold,
            },
            save_path,
        )
        elapsed_h = (time.time() - start) / 3600.0
        print(f"[INFO] Saved {save_path} (best_test_acc={best_acc:.4f}, fold_hours={elapsed_h:.2f})")
        fold_logs.append({"fold": fold, "best_test_acc": best_acc, "hours": elapsed_h})

    # Print summary
    if fold_logs:
        avg = float(np.mean([x["best_test_acc"] for x in fold_logs]))
        print("\n" + "=" * 60)
        print("SUMMARY (best test acc per fold)")
        for x in fold_logs:
            print(f"  Fold {x['fold']}: {x['best_test_acc']:.4f}")
        print(f"  Avg: {avg:.4f}")


if __name__ == "__main__":
    main()


