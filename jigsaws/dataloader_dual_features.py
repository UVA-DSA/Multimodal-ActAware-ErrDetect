"""
Dual-feature dataloader:
- base visual features (from extract_features.py): data/vid_features/{base_model}/{task}/{video}/features.pt
- gesture prompt features (from extract_gesture_prompt_features.py): {gesture_root}/{task}/{video}/features.pt

Used by train_eval_cross_ges_or_ctx_features.py to replace cnn/cnnges with precomputed features.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from dataloader_features import MODEL_FEATURE_DIMS, get_frame_features_in_range
from jigsaws_splits import load_split_records


@dataclass(frozen=True)
class DualFeaturePaths:
    data_root: str = "./data"
    base_model: str = "resnet50"
    gesture_root: str = "./data/gesture_prompt_features"  # points to a run dir inside, typically .../{ckpt_stem}


class JIGSAWS_Gesture_DualFeatures(Dataset):
    """
    Loads per-frame base features and gesture prompt features for each error segment row.

    Returns: (base_features_list[tensor], ges_features_list[tensor], gesture_id_tensor, label_tensor)
    where label is error1_nor0 (0/1) and gesture_id is the numeric gesture (from CSV, e.g. 1,2,3,...)
    """

    def __init__(self, filename: str, task: str, paths: DualFeaturePaths, frame_subsample: int = 1):
        self.filename = filename  # with .csv
        self.task = task
        self.paths = paths
        # Stored JIGSAWS frames are 10 Hz; 2 gives the 5 Hz rate used elsewhere.
        self.frame_subsample = max(1, int(frame_subsample))

        csv_path = os.path.join(paths.data_root, task, "errors", filename)
        self.df = pd.read_csv(csv_path)

        video_name = filename[:-4]
        self.base_path = os.path.join(paths.data_root, "vid_features", paths.base_model, task, video_name, "features.pt")
        self.ges_path = os.path.join(paths.gesture_root, task, video_name, "features.pt")

        self.base_dict: Dict[int, torch.Tensor] = torch.load(self.base_path) if os.path.exists(self.base_path) else {}
        self.ges_dict: Dict[int, torch.Tensor] = torch.load(self.ges_path) if os.path.exists(self.ges_path) else {}

        # Determine dims (best-effort)
        self.base_dim = MODEL_FEATURE_DIMS.get(paths.base_model)
        if self.base_dim is None and len(self.base_dict) > 0:
            self.base_dim = int(next(iter(self.base_dict.values())).shape[0])
        self.ges_dim = int(next(iter(self.ges_dict.values())).shape[0]) if len(self.ges_dict) > 0 else None

        if self.frame_subsample > 1:
            # Subsample on the frames the two streams share, so both stay aligned.
            common = sorted(set(self.base_dict) & set(self.ges_dict))[:: self.frame_subsample]
            keep = set(common)
            self.base_dict = {k: v for k, v in self.base_dict.items() if k in keep}
            self.ges_dict = {k: v for k, v in self.ges_dict.items() if k in keep}
        self._build_sequence()

    @staticmethod
    def _parse_gesture_id(gs: str) -> int:
        gs_str = str(gs)
        if gs_str.startswith("G"):
            return int(gs_str[1:])
        return int(gs_str)

    def _build_sequence(self):
        self.base_seq = []
        self.ges_seq = []
        labels = []
        gestures = []
        for _, row in self.df.iterrows():
            st, et = int(row["start_time"]), int(row["end_time"])
            gs = row["gesture"]
            label = int(row["error1_nor0"])

            base_frames = get_frame_features_in_range(st, et, self.base_dict)
            ges_frames = get_frame_features_in_range(st, et, self.ges_dict)

            base_map = {f: feat for f, feat in base_frames}
            ges_map = {f: feat for f, feat in ges_frames}
            common_frames = sorted(set(base_map.keys()) & set(ges_map.keys()))
            if len(common_frames) == 0:
                continue

            gesture_id = self._parse_gesture_id(gs)
            for f in common_frames:
                self.base_seq.append(base_map[f])
                self.ges_seq.append(ges_map[f])
                labels.append(label)
                gestures.append(gesture_id)

        self.labels_seq = torch.tensor(labels, dtype=torch.long)
        self.gestures_seq = torch.tensor(gestures, dtype=torch.long)

    def __len__(self) -> int:
        return 1 if len(self.base_seq) > 0 else 0

    def __getitem__(self, idx: int):
        if len(self.base_seq) == 0:
            return None
        return self.base_seq, self.ges_seq, self.gestures_seq, self.labels_seq


class JIGSAWS_DualFeatures_SegmentWrapper(Dataset):
    """
    Converts variable-length per-row frame sequences into sliding-window segments.
    """

    def __init__(self, original_dataset: JIGSAWS_Gesture_DualFeatures, segment_length: int = 40, step_size: int = 6):
        self.ds = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.indices: List[Tuple[int, int]] = []

        for idx in range(len(self.ds)):
            item = self.ds[idx]
            if item is None:
                continue
            base_feats, ges_feats, _, _ = item
            n = len(base_feats)
            if n <= 0:
                continue
            if n - segment_length + 1 <= 0:
                self.indices.append((idx, 0))
            else:
                for s in range(0, n - segment_length + 1, step_size):
                    self.indices.append((idx, s))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx, start = self.indices[i]
        item = self.ds[idx]
        if item is None:
            return None
        base_feats, ges_feats, gestures, labels = item
        end = min(start + self.segment_length, len(base_feats))
        base_seg = base_feats[start:end]
        ges_seg = ges_feats[start:end]
        labels_seg = labels[start:end]
        gestures_seg = gestures[start:end]
        if len(base_seg) == 0:
            return None
        return torch.stack(base_seg), torch.stack(ges_seg), labels_seg, gestures_seg


def collate_fn_dual_features_context(batch):
    """
    Collate for per-frame prediction tasks (labels repeated across frames), returning:
      base_features_padded (B,T,Db), ges_features_padded (B,T,Dg),
      masks (B,T) bool, labels_padded (B,T) long, gestures_padded (B,T) long
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    base, ges, labels, gestures = zip(*batch)
    base = [x.to(torch.float32) for x in base]
    ges = [x.to(torch.float32) for x in ges]

    lengths = torch.tensor([x.size(0) for x in base], dtype=torch.long)
    base_padded = pad_sequence(base, batch_first=True, padding_value=0.0)
    ges_padded = pad_sequence(ges, batch_first=True, padding_value=0.0)
    masks = torch.arange(base_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    gestures_padded = pad_sequence(gestures, batch_first=True, padding_value=0)

    return base_padded, ges_padded, masks.to(torch.bool), labels_padded, gestures_padded


def split_selector(repo_root: str, dataset_variant: str, split_root: Optional[str] = None):
    return load_split_records(dataset_variant, split_root=split_root, repo_root=repo_root)


