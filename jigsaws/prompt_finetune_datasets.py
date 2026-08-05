"""
Datasets + collate for prompt finetuning on JIGSAWS segment labels.

We use the existing error CSVs:
  data/{Task}/errors/{VideoName}.csv
with columns: start_time,end_time,gesture,error1_nor0

Supported targets:
- gesture: 8 gesture classes (G1,G2,G3,G4,G5,G6,G8,G9)
- error: binary error / no-error labels from `error1_nor0`
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


GESTURE_IDS_SORTED = [1, 2, 3, 4, 5, 6, 8, 9]
GESTURE_TO_INDEX: Dict[int, int] = {g: i for i, g in enumerate(GESTURE_IDS_SORTED)}
VALID_LABEL_TYPES = {"gesture", "error"}


def _frame_paths_in_range(start_time: int, end_time: int, image_glob: str) -> List[str]:
    frame_pattern = re.compile(r"frame_(\d+)\.png")
    frames = []
    for fpath in glob.glob(image_glob):
        m = frame_pattern.search(os.path.basename(fpath))
        if not m:
            continue
        frame_num = int(m.group(1))
        if start_time <= frame_num <= end_time:
            frames.append((frame_num, fpath))
    frames.sort(key=lambda x: x[0])
    return [p for _, p in frames]


@dataclass(frozen=True)
class PromptDatasetConfig:
    data_root: str = "./data"
    task: str = "Suturing"
    segment_length: int = 40
    step_size: int = 6
    sample_frames: int = 10
    label_type: str = "gesture"


# Backward-compatible alias used by existing imports.
GestureDatasetConfig = PromptDatasetConfig


def _parse_gesture_target(row) -> int:
    gesture_str = str(row["gesture"])  # e.g. "G1"
    gesture_id = int(gesture_str[1:]) if gesture_str.startswith("G") else int(gesture_str)
    return int(GESTURE_TO_INDEX[gesture_id])


def _parse_error_target(row) -> int:
    return int(row["error1_nor0"])


def _parse_target(row, label_type: str) -> int:
    if label_type == "gesture":
        return _parse_gesture_target(row)
    if label_type == "error":
        return _parse_error_target(row)
    raise ValueError(f"Unsupported label_type: {label_type}")


class GestureSegmentDataset(Dataset):
    """
    Each item is a segment: (images_tensor[T,C,H,W], mask[T], target_idx)
    """

    def __init__(
        self,
        video_name: str,
        cfg: PromptDatasetConfig,
        transform=None,
        strict: bool = False,
    ):
        super().__init__()
        self.video_name = video_name  # e.g. Suturing_S02_T01 (no .csv)
        self.cfg = cfg
        self.transform = transform
        self.strict = strict
        self.label_type = str(cfg.label_type).lower()
        if self.label_type not in VALID_LABEL_TYPES:
            raise ValueError(f"Unsupported label_type: {cfg.label_type}. Expected one of {tuple(sorted(VALID_LABEL_TYPES))}")

        csv_path = os.path.join(cfg.data_root, cfg.task, "errors", f"{video_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing error CSV: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.image_glob = os.path.join(cfg.data_root, "vid_frames", cfg.task, video_name, "frame_*.png")

        # Precompute segment indices: list of (row_idx, start_frame_idx_in_range_list)
        self._segments: List[Tuple[int, int]] = []
        self.segment_targets: List[int] = []
        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]
            try:
                target = _parse_target(row, self.label_type)
            except Exception:
                if strict:
                    raise
                continue
            st, et = int(row["start_time"]), int(row["end_time"])
            frame_paths = _frame_paths_in_range(st, et, self.image_glob)
            n = len(frame_paths)
            if n <= 0:
                if strict:
                    raise RuntimeError(f"No frames for {video_name} row {row_idx} range=({st},{et}) glob={self.image_glob}")
                continue
            if n - cfg.segment_length + 1 <= 0:
                self._segments.append((row_idx, 0))
                self.segment_targets.append(target)
            else:
                for s in range(0, n - cfg.segment_length + 1, cfg.step_size):
                    self._segments.append((row_idx, s))
                    self.segment_targets.append(target)

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int):
        row_idx, seg_start = self._segments[idx]
        target_idx = int(self.segment_targets[idx])
        row = self.df.iloc[row_idx]
        st, et = int(row["start_time"]), int(row["end_time"])

        frame_paths = _frame_paths_in_range(st, et, self.image_glob)
        if len(frame_paths) == 0:
            if self.strict:
                raise RuntimeError(f"No frames for segment: {self.video_name} row={row_idx}")
            return None

        seg_end = min(seg_start + self.cfg.segment_length, len(frame_paths))
        seg_paths = frame_paths[seg_start:seg_end]
        if len(seg_paths) == 0:
            return None

        # Uniformly sample a fixed number of frames from the window for speed.
        # This reduces compute from O(segment_length) frames per sample to O(sample_frames).
        k = int(self.cfg.sample_frames)
        if k > 0 and len(seg_paths) > k:
            # linspace indices are deterministic and approximately uniform over the window
            idxs = np.linspace(0, len(seg_paths) - 1, num=k, dtype=int)
            seg_paths = [seg_paths[i] for i in idxs.tolist()]

        imgs = []
        for p in seg_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                if self.strict:
                    raise
                continue
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        if len(imgs) == 0:
            return None

        images = torch.stack(imgs)  # (T,C,H,W)
        mask = torch.ones(images.shape[0], dtype=torch.bool)
        return images, mask, torch.tensor(target_idx, dtype=torch.long)


def collate_gesture_segments(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images, masks, target_idx = zip(*batch)
    images = [x.to(torch.float32) for x in images]
    masks = [m.to(torch.bool) for m in masks]

    images_padded = pad_sequence(images, batch_first=True, padding_value=0.0)  # (B,T,C,H,W)
    # pad_sequence doesn't handle 4D tensors directly; we already have (T,C,H,W) so it will pad on T.
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0).to(torch.bool)

    target_idx = torch.tensor(target_idx, dtype=torch.long)
    return images_padded, masks_padded, target_idx


# Backward-compatible alias used by future generic code.
PromptSegmentDataset = GestureSegmentDataset
collate_prompt_segments = collate_gesture_segments


