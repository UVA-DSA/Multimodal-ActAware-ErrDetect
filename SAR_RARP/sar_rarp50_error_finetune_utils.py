"""
Utilities for SAR-RARP50 ResNet50 error-label finetuning and feature extraction.
"""

from __future__ import annotations

import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_SAR_RARP50_ERROR_FEATURE_DIM = 128


def _extract_number(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def list_video_ids_from_pkl_dir(pkl_dir: str) -> List[str]:
    return sorted(
        [os.path.splitext(name)[0] for name in os.listdir(pkl_dir) if name.endswith(".pkl")],
        key=_extract_number,
    )


def split_video_ids(video_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    items = list(video_ids)
    rnd = random.Random(seed)
    rnd.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    val_ids = items[:n_val]
    train_ids = items[n_val:]
    return train_ids, val_ids


def load_video_data(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_error_labels(pkl_path: str) -> np.ndarray:
    video_data = load_video_data(pkl_path)
    if "error_GT" not in video_data:
        raise KeyError(f"Missing `error_GT` in {pkl_path}")
    return np.asarray(video_data["error_GT"], dtype=np.int64)


def compute_balanced_class_weights(targets: Sequence[int], num_classes: int, device: torch.device) -> torch.Tensor:
    if len(targets) == 0:
        raise ValueError("Cannot compute class weights from an empty target list")
    counts = np.bincount(np.asarray(list(targets), dtype=np.int64), minlength=int(num_classes))
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def sorted_image_paths(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        return []
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    names = [name for name in os.listdir(images_dir) if os.path.splitext(name)[1].lower() in image_exts]
    names.sort(key=_extract_number)
    return [os.path.join(images_dir, name) for name in names]


def resolve_frame_paths(images_dir: str, video_data: Dict[str, Any]) -> List[str]:
    available_paths = sorted_image_paths(images_dir)
    if len(available_paths) == 0:
        return []

    image_names = video_data.get("image_name")
    if image_names is None or len(image_names) == 0:
        return available_paths

    by_name = {os.path.basename(path): path for path in available_paths}
    by_stem = {os.path.splitext(os.path.basename(path))[0]: path for path in available_paths}

    resolved: List[str] = []
    for raw_name in image_names:
        raw_base = os.path.basename(str(raw_name))
        raw_stem = os.path.splitext(raw_base)[0]
        candidate = by_name.get(raw_base) or by_stem.get(raw_stem)
        if candidate is None:
            resolved = []
            break
        resolved.append(candidate)

    return resolved if resolved else available_paths


def get_resnet50_error_transforms(train: bool = False, image_size: int = 224):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if train:
        ops = [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)],
                p=0.6,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                        interpolation=InterpolationMode.BILINEAR,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return transforms.Compose(ops)

    ops = [
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


class SAR_RARP50ErrorResNet(nn.Module):
    """
    Frozen ResNet50 backbone with a lightweight projected error-classification head.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        projected_dim: Optional[int] = DEFAULT_SAR_RARP50_ERROR_FEATURE_DIM,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        from torchvision import models

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        hidden_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.backbone_output_dim = hidden_dim
        self.projected_dim = int(projected_dim) if projected_dim is not None else None
        self.hidden_dim = self.projected_dim if self.projected_dim is not None else hidden_dim
        self.freeze_backbone = bool(freeze_backbone)
        if self.projected_dim is not None:
            self.feature_proj = nn.Sequential(
                nn.Linear(hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.feature_proj = None
        self.classifier = nn.Linear(self.hidden_dim, int(num_classes))

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(images)
        if self.feature_proj is not None:
            hidden = self.feature_proj(hidden)
        logits = self.classifier(hidden)
        return hidden, logits


def build_sar_rarp50_error_model(
    pretrained: bool = True,
    num_classes: int = 2,
    projected_dim: Optional[int] = DEFAULT_SAR_RARP50_ERROR_FEATURE_DIM,
    freeze_backbone: bool = True,
) -> SAR_RARP50ErrorResNet:
    return SAR_RARP50ErrorResNet(
        pretrained=pretrained,
        num_classes=num_classes,
        projected_dim=projected_dim,
        freeze_backbone=freeze_backbone,
    )


def _infer_projected_dim_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    proj_weight = state.get("feature_proj.0.weight")
    if torch.is_tensor(proj_weight) and proj_weight.ndim >= 2:
        return int(proj_weight.size(0))
    return None


def load_sar_rarp50_error_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[SAR_RARP50ErrorResNet, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint must contain `state_dict` or `model_state_dict`: {ckpt_path}")

    meta = ckpt.get("meta", {})
    args = ckpt.get("args", {})
    num_classes = int(meta.get("num_classes") or args.get("num_classes") or 2)
    projected_dim = meta.get("projected_dim")
    if projected_dim is None:
        projected_dim = args.get("projection_dim")
    if projected_dim is None:
        projected_dim = _infer_projected_dim_from_state(state)
    if projected_dim is not None:
        projected_dim = int(projected_dim)

    model = build_sar_rarp50_error_model(
        pretrained=False,
        num_classes=num_classes,
        projected_dim=projected_dim,
        freeze_backbone=bool(meta.get("freeze_backbone", True)),
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    return model, {"meta": meta, "args": args, "ckpt": ckpt}


class SAR_RARP50ErrorFrameDataset(Dataset):
    """
    Flat per-frame dataset built from SAR-RARP50 images and `error_GT` labels.
    """

    def __init__(
        self,
        *,
        data_root: str,
        split_set: str,
        pkl_dir: str,
        video_ids: Optional[Sequence[str]] = None,
        transform=None,
        repeat_factor: int = 1,
        strict: bool = False,
    ):
        self.data_root = data_root
        self.split_set = split_set
        self.pkl_dir = pkl_dir
        self.transform = transform
        self.repeat_factor = max(1, int(repeat_factor))
        self.strict = strict

        all_video_ids = list_video_ids_from_pkl_dir(pkl_dir)
        if video_ids is None:
            self.video_ids = all_video_ids
        else:
            wanted = set(video_ids)
            self.video_ids = [vid for vid in all_video_ids if vid in wanted]

        self.samples: List[Tuple[str, int]] = []
        self.targets: List[int] = []

        for video_id in self.video_ids:
            pkl_path = os.path.join(self.pkl_dir, f"{video_id}.pkl")
            video_data = load_video_data(pkl_path)
            labels = load_error_labels(pkl_path)
            images_dir = os.path.join(self.data_root, self.split_set, video_id, "images")
            frame_paths = resolve_frame_paths(images_dir, video_data)

            n = min(len(frame_paths), int(labels.size))
            if n <= 0:
                if strict:
                    raise RuntimeError(f"No aligned frames for video={video_id} split={self.split_set}")
                continue
            if len(frame_paths) != int(labels.size):
                print(
                    f"[WARN] Length mismatch for {video_id}: "
                    f"frames={len(frame_paths)} labels={int(labels.size)} using first {n}"
                )

            for frame_idx in range(n):
                label = int(labels[frame_idx])
                self.samples.append((frame_paths[frame_idx], label))
                self.targets.append(label)

        self.base_num_samples = len(self.samples)
        if self.repeat_factor > 1 and self.base_num_samples > 0:
            self.samples = self.samples * self.repeat_factor
            self.targets = self.targets * self.repeat_factor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
