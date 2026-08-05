"""
Training utilities for SAR_RARP50 feature-based training.

We keep these separate from the legacy SAR_RARP `util.py` (which was written for
older label/output shapes and gradient accumulation).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score, roc_auc_score
from tqdm import tqdm


def compute_metrics(labels, preds, probs=None) -> Tuple[float, float, float, float]:
    """Returns (f1, balanced_accuracy, jaccard, auc)."""
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    acc = float(balanced_accuracy_score(labels, preds))
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    jac = jaccard_score(labels, preds, average="binary", zero_division=0)
    auc = float("nan")
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float32)
        if labels.size > 0 and np.unique(labels).size >= 2:
            try:
                auc = float(roc_auc_score(labels, probs))
            except Exception:
                auc = float("nan")
    return float(f1), float(acc), float(jac), auc


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary focal loss on logits (Lin et al., 2017), with optional positive-class
    weighting (via pos_weight for consistency with BCEWithLogitsLoss) and optional
    label smoothing.
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


def compute_pos_weight_segment(train_loader, device: torch.device):
    num_pos = 0
    num_neg = 0
    for batch in train_loader:
        if batch is None:
            continue
        _, __, labels, ___ = batch  # features, masks, labels, vids
        labels = labels.view(-1)
        num_pos += int((labels == 1).sum().item())
        num_neg += int((labels == 0).sum().item())
    if num_pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(num_neg) / float(num_pos)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return pos_weight, criterion


def compute_pos_weight_frame(train_loader, device: torch.device):
    """
    Compute pos_weight for frame-level training where labels are (B,T) and masked by masks.
    Expects batch: (features, masks, labels_padded, vids)
    """
    num_pos = 0
    num_neg = 0
    for batch in train_loader:
        if batch is None:
            continue
        _feats, masks, labels, _vids = batch
        masks = masks.to(device).float()
        labels = labels.to(device).float()
        flat_mask = masks.view(-1) > 0
        flat_lab = labels.view(-1)[flat_mask]
        if flat_lab.numel() == 0:
            continue
        num_pos += int((flat_lab == 1).sum().item())
        num_neg += int((flat_lab == 0).sum().item())
    if num_pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(num_neg) / float(num_pos)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return pos_weight, criterion


def compute_pos_weight_frame_from_segment_labels(train_loader, device: torch.device):
    """
    Compute pos_weight for a "frame-style" loss where the dataset provides ONE scalar label per segment,
    and we repeat it across the valid frames (per mask) for loss computation.

    Expects batch: (features, masks, labels_scalar, vids)
    """
    num_pos = 0
    num_neg = 0
    for batch in train_loader:
        if batch is None:
            continue
        _feats, masks, labels, _vids = batch
        masks = masks.to(device).float()  # (B,T) 1 for valid
        labels = labels.to(device).long().view(-1)  # (B,)
        valid_counts = masks.sum(dim=1).long()  # (B,)
        if labels.numel() == 0:
            continue
        num_pos += int(valid_counts[labels == 1].sum().item())
        num_neg += int(valid_counts[labels == 0].sum().item())
    if num_pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(num_neg) / float(num_pos)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return pos_weight, criterion


def train_segment(model, dataloader, criterion, optimizer, device: torch.device, max_grad_norm: float = 1.0):
    model.train()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        feats, masks, labels, _vids = batch
        feats = feats.to(device)
        masks = masks.to(device)
        labels_f = labels.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(feats, masks=masks).squeeze(-1)  # (B,)
        loss = criterion(logits, labels_f)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / max(n_batches, 1)
    return f1, acc, jac, auc, avg_loss


@torch.no_grad()
def eval_segment(model, dataloader, device: torch.device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Eval"):
        if batch is None:
            continue
        feats, masks, labels, _vids = batch
        feats = feats.to(device)
        masks = masks.to(device)
        logits = model(feats, masks=masks).squeeze(-1)  # (B,)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    return f1, acc, jac, auc


def _align_time(outputs: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor):
    """Align masks/labels time dimension to outputs time dimension."""
    t_out = int(outputs.size(1))
    t_in = int(masks.size(1))
    if t_in == t_out:
        return masks, labels
    if t_in > t_out:
        return masks[:, :t_out], labels[:, :t_out]
    pad_len = t_out - t_in
    mask_pad = torch.zeros(masks.size(0), pad_len, device=masks.device, dtype=masks.dtype)
    lab_pad = torch.zeros(labels.size(0), pad_len, device=labels.device, dtype=labels.dtype)
    return torch.cat([masks, mask_pad], dim=1), torch.cat([labels, lab_pad], dim=1)


def compute_pos_weight_context(train_loader, device: torch.device):
    num_pos = 0
    num_neg = 0
    for batch in train_loader:
        if batch is None:
            continue
        _base, _ges, masks, labels, _vids = batch
        masks = masks.to(device).float()
        labels = labels.to(device).float()
        # flatten with mask
        flat_mask = masks.view(-1) > 0
        flat_lab = labels.view(-1)[flat_mask]
        if flat_lab.numel() == 0:
            continue
        num_pos += int((flat_lab == 1).sum().item())
        num_neg += int((flat_lab == 0).sum().item())
    if num_pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(num_neg) / float(num_pos)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return pos_weight, criterion


def train_frame(model, dataloader, criterion, optimizer, device: torch.device, max_grad_norm: float = 1.0):
    """
    Frame-level training for single-stream model outputting logits (B,T_out).
    Expects batch: (features, masks, labels_padded, vids)
    """
    model.train()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        feats, masks, labels, _vids = batch
        feats = feats.to(device)
        masks_f = masks.to(device).float()
        labels_f = labels.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(feats, masks=masks.to(device)).to(torch.float32)  # (B,T_out)
        masks_f, labels_f = _align_time(logits, masks_f, labels_f)

        flat_mask = masks_f.view(-1) > 0
        flat_logits = logits.view(-1)[flat_mask]
        flat_labels = labels_f.view(-1)[flat_mask]
        if flat_labels.numel() == 0:
            continue

        loss = criterion(flat_logits, flat_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(flat_logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(flat_labels.detach().cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / max(n_batches, 1)
    return f1, acc, jac, auc, avg_loss


@torch.no_grad()
def eval_frame(model, dataloader, device: torch.device):
    """
    Frame-level eval for single-stream model outputting logits (B,T_out).
    Expects batch: (features, masks, labels_padded, vids)
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Eval"):
        if batch is None:
            continue
        feats, masks, labels, _vids = batch
        feats = feats.to(device)
        masks_f = masks.to(device).float()
        labels_f = labels.to(device).float()

        logits = model(feats, masks=masks.to(device)).to(torch.float32)
        masks_f, labels_f = _align_time(logits, masks_f, labels_f)

        flat_mask = masks_f.view(-1) > 0
        flat_logits = logits.view(-1)[flat_mask]
        flat_labels = labels_f.view(-1)[flat_mask]
        if flat_labels.numel() == 0:
            continue

        probs = torch.sigmoid(flat_logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(flat_labels.cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    return f1, acc, jac, auc


def train_frame_from_segment(model, dataloader, criterion, optimizer, device: torch.device, max_grad_norm: float = 1.0):
    """
    Frame-style training where:
      - model outputs ONE logit per segment/window (B,1) or (B,)
      - dataset provides ONE scalar label per segment/window (B,)
    We repeat both across the valid frames (per mask) and compute loss/metrics over frames.

    Expects batch: (features, masks, labels_scalar, vids)
    """
    model.train()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        feats, masks, labels, _vids = batch
        feats = feats.to(device)
        masks_f = masks.to(device).float()  # (B,T)
        labels_f = labels.to(device).float().view(-1)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        logits_seg = model(feats, masks=masks.to(device)).to(torch.float32).view(-1)  # (B,)

        # Repeat to frame shape
        t = int(masks_f.size(1))
        logits = logits_seg.unsqueeze(1).expand(-1, t)  # (B,T)
        labels_rep = labels_f.unsqueeze(1).expand(-1, t)  # (B,T)

        flat_mask = masks_f.reshape(-1) > 0
        flat_logits = logits.reshape(-1)[flat_mask]
        flat_labels = labels_rep.reshape(-1)[flat_mask]
        if flat_labels.numel() == 0:
            continue

        loss = criterion(flat_logits, flat_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(flat_logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(flat_labels.detach().cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / max(n_batches, 1)
    return f1, acc, jac, auc, avg_loss


@torch.no_grad()
def eval_frame_from_segment(model, dataloader, device: torch.device):
    """
    Frame-style eval where we repeat the segment label and segment logit across valid frames.
    Expects batch: (features, masks, labels_scalar, vids)
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Eval"):
        if batch is None:
            continue
        feats, masks, labels, _vids = batch
        feats = feats.to(device)
        masks_f = masks.to(device).float()
        labels_f = labels.to(device).float().view(-1)

        logits_seg = model(feats, masks=masks.to(device)).to(torch.float32).view(-1)  # (B,)
        t = int(masks_f.size(1))
        logits = logits_seg.unsqueeze(1).expand(-1, t)  # (B,T)
        labels_rep = labels_f.unsqueeze(1).expand(-1, t)  # (B,T)

        flat_mask = masks_f.reshape(-1) > 0
        flat_logits = logits.reshape(-1)[flat_mask]
        flat_labels = labels_rep.reshape(-1)[flat_mask]
        if flat_labels.numel() == 0:
            continue

        probs = torch.sigmoid(flat_logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(flat_labels.cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    return f1, acc, jac, auc

def train_context_dual(model, dataloader, criterion, optimizer, device: torch.device, max_grad_norm: float = 1.0):
    model.train()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        base, ges, masks, labels, _vids = batch
        base = base.to(device)
        ges = ges.to(device)
        masks_f = masks.to(device).float()
        labels_f = labels.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(base, ges, masks=masks.to(device)).to(torch.float32)  # (B,T_out)
        masks_f, labels_f = _align_time(logits, masks_f, labels_f)

        flat_mask = masks_f.view(-1) > 0
        flat_logits = logits.view(-1)[flat_mask]
        flat_labels = labels_f.view(-1)[flat_mask]
        if flat_labels.numel() == 0:
            continue

        loss = criterion(flat_logits, flat_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(flat_logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(flat_labels.detach().cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / max(n_batches, 1)
    return f1, acc, jac, auc, avg_loss


@torch.no_grad()
def eval_context_dual(model, dataloader, device: torch.device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Eval"):
        if batch is None:
            continue
        base, ges, masks, labels, _vids = batch
        base = base.to(device)
        ges = ges.to(device)
        masks_f = masks.to(device).float()
        labels_f = labels.to(device).float()

        logits = model(base, ges, masks=masks.to(device)).to(torch.float32)
        masks_f, labels_f = _align_time(logits, masks_f, labels_f)

        flat_mask = masks_f.view(-1) > 0
        flat_logits = logits.view(-1)[flat_mask]
        flat_labels = labels_f.view(-1)[flat_mask]
        if flat_labels.numel() == 0:
            continue
        probs = torch.sigmoid(flat_logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(flat_labels.cpu().numpy().astype(int).tolist())

    f1, acc, jac, auc = compute_metrics(all_labels, all_preds, all_probs)
    return f1, acc, jac, auc


