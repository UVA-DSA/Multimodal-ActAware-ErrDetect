"""
Utility functions for training with pre-extracted features.

These are the same as util.py but renamed for clarity when using feature-based training.

Classification metrics use sklearn's balanced_accuracy_score where "accuracy" is reported
(train_accuracy / test_accuracy in logs).
"""

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score, confusion_matrix
from tqdm import tqdm
import numpy as np


def window_labels_from_frames(
    labels: torch.Tensor,
    masks: torch.Tensor,
    window_label_rule: str = "majority",
) -> torch.Tensor:
    """
    Aggregate per-frame labels into one label per window (batch row).

    Args:
        labels: (B, T) numeric (0/1 expected).
        masks: (B, T) bool/float; valid frames where mask > 0.
        window_label_rule:
            - "majority": count 1s vs 0s on valid frames; ties resolve to 1 (error).
            - "any_error": 1 if any valid frame has label 1; windows with no valid frames are 0.

    Returns:
        (B,) long
    """
    if window_label_rule not in ("majority", "any_error"):
        raise ValueError(
            f"window_label_rule must be 'majority' or 'any_error', got {window_label_rule!r}"
        )
    valid = masks > 0
    if window_label_rule == "majority":
        ones = ((labels == 1) & valid).sum(dim=1)
        zeros = ((labels == 0) & valid).sum(dim=1)
        return (ones >= zeros).long()
    return (((labels == 1) & valid).any(dim=1)).long()


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary focal loss on logits (Lin et al., 2017), with optional positive-class
    weighting (alpha implemented via pos_weight for consistency with BCEWithLogitsLoss)
    and optional label smoothing.
    """

    def __init__(self, gamma: float = 2.0, pos_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight))
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = bce * (1.0 - p_t).clamp(min=1e-6).pow(self.gamma)
        if self.pos_weight is not None:
            weight = targets * self.pos_weight.to(logits.device) + (1.0 - targets)
            loss = loss * weight
        return loss.mean()


class SmoothedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing applied to binary targets."""

    def __init__(self, pos_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight))
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def build_criterion(loss: str = "wbce", pos_weight=None, focal_gamma: float = 2.0, label_smoothing: float = 0.0):
    """
    Build the training criterion.

    Args:
        loss: "wbce" (BCEWithLogits, optionally class-weighted; the paper setting) or "focal".
        pos_weight: optional scalar/tensor weight for the positive class.
        focal_gamma: gamma for focal loss.
        label_smoothing: label smoothing epsilon in [0, 0.5).
    """
    if loss == "focal":
        return BinaryFocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight, label_smoothing=label_smoothing)
    if loss != "wbce":
        raise ValueError(f"Unknown loss: {loss!r} (expected 'wbce' or 'focal')")
    if label_smoothing > 0:
        return SmoothedBCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=label_smoothing)
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight))
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def print_distribution(labels, preds, prefix=""):
    """Print the distribution of labels and predictions."""
    label_counts = {0: (labels == 0).sum(), 1: (labels == 1).sum()}
    pred_counts = {0: (preds == 0).sum(), 1: (preds == 1).sum()}
    
    total = len(labels)
    print(f"  {prefix}Labels:      0={label_counts[0]:4d} ({100*label_counts[0]/total:.1f}%), 1={label_counts[1]:4d} ({100*label_counts[1]/total:.1f}%)")
    print(f"  {prefix}Predictions: 0={pred_counts[0]:4d} ({100*pred_counts[0]/total:.1f}%), 1={pred_counts[1]:4d} ({100*pred_counts[1]/total:.1f}%)")


def compute_metrics(labels, preds):
    """
    Compute metrics with proper handling of edge cases.

    The second return value is balanced accuracy (sklearn balanced_accuracy_score), not plain accuracy.

    Returns: (f1, balanced_accuracy, jaccard)
    """
    # Ensure integer arrays
    labels = np.array(labels).astype(int)
    preds = np.array(preds).astype(int)

    accuracy = float(balanced_accuracy_score(labels, preds))
    
    # F1 score - use zero_division=0 to handle cases where a class is not predicted
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    
    # Jaccard score - use zero_division=0 to handle edge cases
    jaccard = jaccard_score(labels, preds, average='binary', zero_division=0)
    
    return f1, accuracy, jaccard


def compute_pos_weight(train_loader, device, window_label_rule: str = "majority"):
    """Compute positive class weight for imbalanced classification."""
    num_positive_samples = 0
    num_negative_samples = 0

    for batch in train_loader:
        if batch is None:
            continue
        # Unpack with masks (4 elements now)
        _, masks, labels, _ = batch
        window_labels = window_labels_from_frames(labels, masks, window_label_rule)
        num_positive_samples += (window_labels == 1).sum().item()
        num_negative_samples += (window_labels == 0).sum().item()

    if num_positive_samples == 0:
        print("[WARN] compute_pos_weight: No positive samples found; defaulting pos_weight to 1.0")
        pos_weight_value = 1.0
    else:
        pos_weight_value = num_negative_samples / num_positive_samples
        
    print(f"[INFO] Class distribution: negative={num_negative_samples}, positive={num_positive_samples}")
    print(f"[INFO] pos_weight = {pos_weight_value:.2f}")
    
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return pos_weight, criterion


def compute_pos_weight_context(train_loader, device, window_label_rule: str = "majority"):
    """
    Compute pos_weight for per-frame context-style training where labels are repeated across frames
    and masked by masks.
    Supports both single-feature and dual-feature loaders.
    """
    num_positive = 0
    num_negative = 0

    for batch in train_loader:
        if batch is None:
            continue
        if len(batch) == 4:
            # (features, masks, labels, gestures)
            _, masks, labels, _ = batch
        else:
            # (base_features, ges_features, masks, labels, gestures)
            _, __, masks, labels, _ = batch

        window_labels = window_labels_from_frames(labels, masks, window_label_rule)
        num_positive += (window_labels == 1).sum().item()
        num_negative += (window_labels == 0).sum().item()

    if num_positive == 0:
        print("[WARN] compute_pos_weight_context: No positive samples found; defaulting pos_weight to 1.0")
        pos_weight_value = 1.0
    else:
        pos_weight_value = num_negative / num_positive

    print(f"[INFO] Context label distribution (masked frames): negative={num_negative}, positive={num_positive}")
    print(f"[INFO] pos_weight = {pos_weight_value:.2f}")
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return pos_weight, criterion


def _align_time_dim_to_outputs(outputs: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor):
    """
    Ensure masks/labels have the same time dimension as model outputs.

    Some models pad/truncate internally to n_frames (e.g. 40), while collate_fn pads only
    to the max sequence length in the batch (which can be < n_frames). This mismatch can
    break masked indexing like flat_out[flat_mask].
    """
    t_out = int(outputs.size(1))
    t_in = int(masks.size(1))
    if t_in == t_out:
        return masks, labels
    if t_in > t_out:
        return masks[:, :t_out], labels[:, :t_out]

    pad_len = t_out - t_in
    mask_pad = torch.zeros(masks.size(0), pad_len, device=masks.device, dtype=masks.dtype)
    label_pad = torch.zeros(labels.size(0), pad_len, device=labels.device, dtype=labels.dtype)
    masks = torch.cat([masks, mask_pad], dim=1)
    labels = torch.cat([labels, label_pad], dim=1)
    return masks, labels


def _dominant_window_labels(labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias: majority rule (tie -> error)."""
    return window_labels_from_frames(labels, masks, "majority")


def train_context_dual_features(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_grad_norm: float = 1.0,
    window_label_rule: str = "majority",
):
    """
    Per-frame training for dual-feature context model.
    Expects batch: (base_features, ges_features, masks, labels, gestures)
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
        base_feats, ges_feats, masks, labels, _ = batch
        base_feats = base_feats.to(device)
        ges_feats = ges_feats.to(device)
        masks = masks.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(base_feats, ges_feats, masks=masks.to(torch.bool))  # (B,T_out)

        # Align masks/labels time dim to model output time dim (e.g. 34 -> 40)
        masks, labels = _align_time_dim_to_outputs(outputs, masks, labels)
        window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()
        valid_windows = masks.sum(dim=1) > 0
        if valid_windows.sum().item() == 0:
            continue
        pooled_logits = (outputs * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
        filt_out = pooled_logits[valid_windows]
        filt_lab = window_labels[valid_windows]

        loss = criterion(filt_out, filt_lab)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(filt_out).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(filt_lab.detach().cpu().numpy().astype(int).tolist())

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
def test_context_dual_features(
    model,
    dataloader,
    device,
    debug: bool = False,
    criterion: nn.Module | None = None,
    window_label_rule: str = "majority",
):
    """
    Per-frame evaluation for dual-feature context model.
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
        base_feats, ges_feats, masks, labels, _ = batch
        base_feats = base_feats.to(device)
        ges_feats = ges_feats.to(device)
        masks = masks.to(device).float()
        labels = labels.to(device).float()

        outputs = model(base_feats, ges_feats, masks=masks.to(torch.bool))

        # Align masks/labels time dim to model output time dim
        masks, labels = _align_time_dim_to_outputs(outputs, masks, labels)
        window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()
        valid_windows = masks.sum(dim=1) > 0
        if valid_windows.sum().item() == 0:
            continue
        pooled_logits = (outputs * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
        filt_out = pooled_logits[valid_windows]
        filt_lab = window_labels[valid_windows]

        if debug and not debug_printed:
            print("  [DEBUG] base_feats", tuple(base_feats.shape), "ges_feats", tuple(ges_feats.shape), "masks", tuple(masks.shape), "outputs", tuple(outputs.shape))
            debug_printed = True

        if criterion is not None:
            loss = criterion(filt_out, filt_lab)
            total_loss += loss.item()
            num_batches += 1

        probs = torch.sigmoid(filt_out).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(filt_lab.cpu().numpy().astype(int).tolist())

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

def train_features(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_grad_norm=1.0,
    window_label_rule: str = "majority",
):
    """
    Training function for feature-based models.
    
    Uses one label per segment (not repeated across frames).
    No gradient accumulation - updates weights every batch.
    
    Args:
        model: The model (GVRModulePredFeatures or similar)
        dataloader: DataLoader returning (features, masks, labels, gestures)
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        max_grad_norm: Maximum gradient norm for clipping (prevents large updates)
    """
    model.train()
    all_preds = []
    all_labels = []
    all_probs = []  # Store raw probabilities for analysis
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        
        features, masks, labels, gestures = batch
        features = features.to(device)
        masks = masks.to(device)  # bool mask: [batch_size, seq_len]
        labels = labels.to(device)
        window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with pre-extracted features and attention masks
        outputs = model(features, masks=masks)  # Shape: [batch_size, 1]
        outputs = outputs.squeeze(-1)  # Shape: [batch_size]

        # Compute loss - one prediction vs one label per segment
        loss = criterion(outputs, window_labels)
        
        # Backpropagate
        loss.backward()
        
        # Gradient clipping to prevent large updates that cause oscillation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Collect predictions, probabilities, and labels
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(window_labels.cpu().numpy().tolist())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)

    # Print statistics
    avg_loss = total_loss / max(num_batches, 1)
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Prob stats: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, std={all_probs.std():.3f}")
    print_distribution(all_labels, all_preds, prefix="Train ")

    # Calculate evaluation metrics
    f1, accuracy, jaccard = compute_metrics(all_labels, all_preds)

    return f1, accuracy, jaccard


def test_features(
    model,
    dataloader,
    device,
    debug=False,
    criterion: nn.Module | None = None,
    window_label_rule: str = "majority",
):
    """
    Testing function for feature-based models.
    
    Uses one label per segment (not repeated across frames).
    
    Args:
        model: The model (GVRModulePredFeatures or similar)
        dataloader: DataLoader returning (features, masks, labels, gestures)
        device: torch device
    """
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    debug_printed = False
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if batch is None:
                continue
            
            features, masks, labels, gestures = batch
            features = features.to(device)
            masks = masks.to(device)  # bool mask: [batch_size, seq_len]
            labels = labels.to(device)
            window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()

            # Forward pass with pre-extracted features and attention masks
            outputs = model(features, masks=masks)  # Shape: [batch_size, 1]
            outputs = outputs.squeeze(-1)  # Shape: [batch_size]

            if debug and not debug_printed:
                # Feature/Mask/Logit diagnostics for the first non-empty batch
                with torch.no_grad():
                    # Per-sample average frame embedding norm (pre-projection)
                    per_sample_frame_norm = features.float().norm(dim=-1)  # (B, T)
                    per_sample_mean_norm = per_sample_frame_norm.masked_fill(~masks, 0.0).sum(dim=1) / masks.sum(dim=1).clamp_min(1)
                    print("  [DEBUG] Eval batch shapes:",
                          f"features={tuple(features.shape)}, masks={tuple(masks.shape)}, outputs={tuple(outputs.shape)}")
                    print("  [DEBUG] Eval mask valid frames:",
                          f"min={int(masks.sum(dim=1).min().item())}, max={int(masks.sum(dim=1).max().item())}")
                    print("  [DEBUG] Eval feature mean-norm per sample:",
                          f"min={per_sample_mean_norm.min().item():.6f}, max={per_sample_mean_norm.max().item():.6f}, std={per_sample_mean_norm.std().item():.6f}")
                    print("  [DEBUG] Eval logits stats:",
                          f"min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}, std={outputs.std().item():.6f}")
                debug_printed = True

            if criterion is not None:
                loss = criterion(outputs, window_labels)
                total_loss += loss.item()
                num_batches += 1

            # Collect predictions, probabilities, and labels
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(window_labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)

    # Print statistics
    if criterion is not None:
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Prob stats: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, std={all_probs.std():.3f}")
    print_distribution(all_labels, all_preds, prefix="Test  ")

    # Calculate evaluation metrics
    f1, accuracy, jaccard = compute_metrics(all_labels, all_preds)
    
    return f1, accuracy, jaccard


def train_context_features(model, dataloader, criterion, optimizer, device, window_label_rule: str = "majority"):
    """
    Training function for context-based feature models (per-frame prediction).
    
    This version still uses per-frame labels since the model predicts per-frame.
    No gradient accumulation.
    
    Args:
        model: The model (GVRModuleContexPredFeatures or similar)
        dataloader: DataLoader returning (features, masks, labels, gestures)
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
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
        
        features, masks, labels, gestures = batch
        features = features.to(device)
        masks = masks.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass - returns per-frame predictions
        outputs = model(features)  # Shape: [batch, n_frames]
        masks, labels = _align_time_dim_to_outputs(outputs, masks, labels)
        window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()
        valid_windows = masks.sum(dim=1) > 0
        if valid_windows.sum().item() == 0:
            continue
        pooled_logits = (outputs * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
        filtered_outputs = pooled_logits[valid_windows]
        filtered_labels = window_labels[valid_windows]

        loss = criterion(filtered_outputs, filtered_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(filtered_outputs).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(filtered_labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)

    # Print statistics
    avg_loss = total_loss / max(num_batches, 1)
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Prob stats: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, std={all_probs.std():.3f}")
    print_distribution(all_labels, all_preds, prefix="Train ")

    # Calculate evaluation metrics
    f1, accuracy, jaccard = compute_metrics(all_labels, all_preds)

    return f1, accuracy, jaccard


def test_context_features(
    model,
    dataloader,
    device,
    criterion: nn.Module | None = None,
    window_label_rule: str = "majority",
):
    """
    Testing function for context-based feature models (per-frame prediction).
    
    Args:
        model: The model (GVRModuleContexPredFeatures or similar)
        dataloader: DataLoader returning (features, masks, labels, gestures)
        device: torch device
    """
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if batch is None:
                continue
            
            features, masks, labels, gestures = batch
            features = features.to(device)
            masks = masks.to(device).float()
            labels = labels.to(device)

            outputs = model(features)  # Shape: [batch, n_frames]
            masks, labels = _align_time_dim_to_outputs(outputs, masks, labels)
            window_labels = window_labels_from_frames(labels, masks, window_label_rule).float()
            valid_windows = masks.sum(dim=1) > 0
            if valid_windows.sum().item() == 0:
                continue
            pooled_logits = (outputs * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
            filtered_outputs = pooled_logits[valid_windows]
            filtered_labels = window_labels[valid_windows]
                
            if criterion is not None:
                loss = criterion(filtered_outputs, filtered_labels)
                total_loss += loss.item()
                num_batches += 1

            probs = torch.sigmoid(filtered_outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(filtered_labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)

    # Print statistics
    if criterion is not None:
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Prob stats: min={all_probs.min():.3f}, max={all_probs.max():.3f}, mean={all_probs.mean():.3f}, std={all_probs.std():.3f}")
    print_distribution(all_labels, all_preds, prefix="Test  ")

    # Calculate evaluation metrics
    f1, accuracy, jaccard = compute_metrics(all_labels, all_preds)
    
    return f1, accuracy, jaccard
