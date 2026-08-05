"""
Utility functions for training/evaluating feature+kinematics models.

Mirrors the "enhanced" behavior from `util_features.py`:
- robust metric computation
- probability distribution stats
- optional pos_weight for imbalance
- gradient clipping support
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from util_features import compute_metrics, print_distribution, window_labels_from_frames


def compute_pos_weight_kin_features(train_loader, device, window_label_rule: str = "majority"):
    """Compute pos_weight for segment-level classification with (features,kine,masks,labels,gestures)."""
    num_positive = 0
    num_negative = 0

    for batch in train_loader:
        if batch is None:
            continue
        _, __, masks, labels, ____ = batch
        window_labels = window_labels_from_frames(labels, masks, window_label_rule)
        num_positive += (window_labels == 1).sum().item()
        num_negative += (window_labels == 0).sum().item()

    if num_positive == 0:
        print("[WARN] compute_pos_weight_kin_features: No positive samples found; defaulting pos_weight to 1.0")
        pos_weight_value = 1.0
    else:
        pos_weight_value = num_negative / num_positive

    print(f"[INFO] Class distribution: negative={num_negative}, positive={num_positive}")
    print(f"[INFO] pos_weight = {pos_weight_value:.2f}")

    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return pos_weight, criterion


def train_kin_features(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_grad_norm: float = 1.0,
    window_label_rule: str = "majority",
):
    """
    Train for one epoch.
    Expects batch: (features, kine, masks, labels, gestures)
    """
    model.train()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        features, kine, masks, labels, _ = batch
        features = features.to(device)
        kine = kine.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(features, kine, masks=masks).squeeze(-1)  # (B,)
        loss = criterion(outputs, window_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(window_labels.detach().cpu().numpy().astype(int).tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs) if len(all_probs) else np.array([0.0])

    avg_loss = total_loss / max(num_batches, 1)
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Prob stats: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, std={all_probs.std():.3f}")
    print_distribution(all_labels, all_preds, prefix="Train ")

    f1, accuracy, jaccard = compute_metrics(all_labels, all_preds)
    return f1, accuracy, jaccard


@torch.no_grad()
def test_kin_features(
    model,
    dataloader,
    device,
    debug: bool = False,
    criterion: nn.Module | None = None,
    window_label_rule: str = "majority",
):
    """
    Eval for one epoch.
    Expects batch: (features, kine, masks, labels, gestures)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    debug_printed = False
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Testing"):
        if batch is None:
            continue
        features, kine, masks, labels, _ = batch
        features = features.to(device)
        kine = kine.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()

        outputs = model(features, kine, masks=masks).squeeze(-1)  # (B,)

        if debug and not debug_printed:
            print("  [DEBUG] Eval batch shapes:",
                  f"features={tuple(features.shape)}, kine={tuple(kine.shape)}, masks={tuple(masks.shape)}, outputs={tuple(outputs.shape)}")
            debug_printed = True

        if criterion is not None:
            loss = criterion(outputs, window_labels)
            total_loss += loss.item()
            num_batches += 1

        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(window_labels.cpu().numpy().astype(int).tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs) if len(all_probs) else np.array([0.0])

    if criterion is not None:
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Prob stats: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, std={all_probs.std():.3f}")
    print_distribution(all_labels, all_preds, prefix="Test  ")
    f1, accuracy, jaccard = compute_metrics(all_labels, all_preds)
    return f1, accuracy, jaccard

