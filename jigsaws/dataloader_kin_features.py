"""
Feature + Kinematics Dataloader for JIGSAWS
------------------------------------------

Loads:
- Pre-extracted visual features (from `extract_features.py`) per frame
- Kinematics per frame (standardized per video, optionally downsampled)

This is the kinematics analogue of `dataloader_features.py`.

Expected feature path:
  default: {data_root}/vid_features/{model_name}/{task}/{video_name}/features.pt
  custom:  {feature_root}/{task}/{video_name}/features.pt

Expected kinematics path (same as `dataloader_kin.py`):
  {data_root}/{task}/kinematics/{video_name}

Expected error annotation path:
  {data_root}/{task}/errors/{filename}

Timing note:
- Visual frames under `data/vid_frames/...` are already downsampled from 30 Hz
  to 10 Hz, but preserve original frame numbering (`0, 3, 6, ...`).
- Error annotations are expressed in that same original frame-number space.
- Kinematics therefore need to be downsampled to the same preserved indices to
  align with the visual features.
"""

from __future__ import annotations

import os
import random
import re
from typing import Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

from dataloader_features import (
    MODEL_FEATURE_DIMS,
    get_feature_dim,
    get_frame_features_in_range,
    infer_feature_dim_from_root,
    resolve_feature_path,
)
from jigsaws_splits import build_dataset_variant_map, load_split_records, parse_dataset_variant


KINEMATICS_COLUMNS = [
    # PSML (left) position & velocity
    "PSML_position_x", "PSML_position_y", "PSML_position_z",
    "PSML_velocity_x", "PSML_velocity_y", "PSML_velocity_z",
    # PSML gripper
    "PSML_gripper_angle",
    # PSMR (right) position & velocity
    "PSMR_position_x", "PSMR_position_y", "PSMR_position_z",
    "PSMR_velocity_x", "PSMR_velocity_y", "PSMR_velocity_z",
    # PSMR gripper
    "PSMR_gripper_angle",
]


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each column (z-score) with numerical safety."""
    col_means = df.mean(axis=0)
    col_stds = df.std(axis=0).replace(0, 1.0)
    return (df - col_means) / col_stds


def _kin_debug_stats(arr: np.ndarray) -> dict:
    """Compute compact stats for debug printing. Expects float array."""
    finite = np.isfinite(arr)
    if not finite.any():
        return {"finite": 0, "total": int(arr.size)}
    a = arr[finite]
    return {
        "finite": int(finite.sum()),
        "total": int(arr.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "zero_frac": float(np.mean(a == 0.0)),
        "abs_mean": float(np.mean(np.abs(a))),
    }


def _print_kin_debug(df: pd.DataFrame, path: str, stage: str):
    """Print kinematics health checks (nonzero + normalization)."""
    arr = df.to_numpy(dtype=np.float32, copy=True)
    stats = _kin_debug_stats(arr)
    print(f"[KIN][{stage}] file={path}")
    if stats.get("finite", 0) == 0:
        print("  [KIN][WARN] all values are non-finite (NaN/Inf).")
        return
    print(
        "  [KIN] shape=", tuple(arr.shape),
        "min=", f"{stats['min']:.4f}",
        "max=", f"{stats['max']:.4f}",
        "mean=", f"{stats['mean']:.4f}",
        "std=", f"{stats['std']:.4f}",
        "zero_frac=", f"{stats['zero_frac']:.4f}",
    )
    # Per-column normalization sanity (after z-scoring, should be near mean~0, std~1)
    col_means = df.mean(axis=0).to_numpy()
    col_stds = df.std(axis=0).to_numpy()
    if np.isfinite(col_means).all() and np.isfinite(col_stds).all():
        print(
            "  [KIN] per-col |mean|: max=", f"{np.max(np.abs(col_means)):.4f}",
            "per-col std: min=", f"{np.min(col_stds):.4f}",
            "max=", f"{np.max(col_stds):.4f}",
        )
        # Flag obvious issues
        if np.max(np.abs(col_means)) > 0.5 or np.max(np.abs(col_stds - 1.0)) > 0.5:
            print("  [KIN][WARN] normalization looks off (expected per-col mean~0, std~1).")
    else:
        print("  [KIN][WARN] per-column mean/std contains NaN/Inf.")


class JIGSAWS_Gesture_Kin_Features(Dataset):
    """
    Returns raw (variable-length) frame sequences for a gesture segment:
      - features: list[Tensor(D)]
      - kine: Tensor(T, K)
      - gs: Tensor()
      - label: Tensor()
    """

    def __init__(
        self,
        filename: str,
        task: str,
        model_name: str = "resnet50",
        data_root: str = "./data",
        feature_root: str | None = None,
        kin_downsample: int = 3,
        frame_subsample: int = 1,
        debug_kin: bool = False,
    ):
        self.filename = filename
        self.task = task
        self.model_name = model_name
        self.data_root = data_root
        self.feature_root = feature_root
        self.kin_downsample = kin_downsample
        self.frame_subsample = max(1, int(frame_subsample))
        self.debug_kin = debug_kin or (os.environ.get("KIN_DEBUG", "0") == "1")

        self.error_path = f"{data_root}/{task}/errors/{filename}"
        self.curtb = pd.read_csv(self.error_path)
        self.label = self.curtb["error1_nor0"]

        # Kinematics path varies across setups; try a few common conventions.
        video_name = filename[:-4]
        kin_candidates = [
            f"{data_root}/{task}/kinematics/{video_name}",   # matches `dataloader_kin.py`
            f"{data_root}/{task}/kinematics/{filename}",     # some setups keep the .csv name
            f"{data_root}/{task}/kinematics/{video_name}.csv",
            f"{data_root}/{task}/kinematics/{filename}.csv",
        ]
        self.kinematics_path = next((p for p in kin_candidates if os.path.exists(p)), None)
        if self.kinematics_path is None:
            raise FileNotFoundError(
                "Could not find kinematics file. Tried:\n  - " + "\n  - ".join(kin_candidates)
            )

        self.kine = pd.read_csv(self.kinematics_path)
        if self.kine.empty:
            raise ValueError(f"Kinematics file is empty: {self.kinematics_path}")
        if self.debug_kin:
            _print_kin_debug(self.kine, self.kinematics_path, stage="raw_loaded")

        self.kine = _standardize_df(self.kine)
        if self.kin_downsample and self.kin_downsample > 1:
            # Keep original row indices so .loc[frame_num] still works when
            # visual features use preserved original frame ids like 0, 3, 6, ...
            self.kine = self.kine.iloc[:: self.kin_downsample]
        self.kine = self.kine[KINEMATICS_COLUMNS]
        self.kine_dim = len(KINEMATICS_COLUMNS)
        if self.debug_kin:
            _print_kin_debug(self.kine, self.kinematics_path, stage=f"standardized_ds{self.kin_downsample}_selectedcols")

        # Visual features
        self.feature_dim = MODEL_FEATURE_DIMS.get(model_name)
        if self.feature_dim is None and feature_root is None:
            self.feature_dim = get_feature_dim(model_name)
        self.features_path = resolve_feature_path(
            task,
            video_name,
            data_root=data_root,
            model_name=model_name,
            feature_root=feature_root,
        )
        if not os.path.exists(self.features_path):
            print(f"[WARN] Features file not found: {self.features_path}")
            if feature_root is not None:
                print("[WARN] Check that --feature_root points to a directory shaped as {root}/{task}/{video}/features.pt")
            else:
                print(f"[WARN] Run: python extract_features.py --model {model_name}")
            self.features_dict: dict[int, torch.Tensor] = {}
        else:
            self.features_dict = torch.load(self.features_path, map_location="cpu")
            if len(self.features_dict) > 0:
                sample_feat = next(iter(self.features_dict.values()))
                actual_dim = int(sample_feat.shape[0])
                if self.feature_dim is not None and actual_dim != self.feature_dim:
                    print(f"[WARN] Feature dim mismatch: expected {self.feature_dim}, got {actual_dim}")
                self.feature_dim = actual_dim
            print(f"[INFO] Loaded {len(self.features_dict)} frame features (dim={self.feature_dim}) from {self.features_path}")

        # Optionally subsample stored frames (10 Hz on disk); kinematics stay aligned
        # because they are looked up by preserved original frame number below.
        if self.frame_subsample > 1 and len(self.features_dict) > 0:
            kept = sorted(self.features_dict.keys())[:: self.frame_subsample]
            self.features_dict = {k: self.features_dict[k] for k in kept}

        self._missing_kin_warned = False
        self._missing_kin_count = 0
        self._missing_kin_total = 0
        self._build_sequence()

    @staticmethod
    def _parse_gesture_id(gs):
        gs_str = str(gs)
        if gs_str.startswith("G"):
            return int(gs_str[1:])
        return int(gs_str)

    def _build_sequence(self):
        self.features_seq = []
        kine_rows = []
        labels = []
        gestures = []
        for _, row in self.curtb.iterrows():
            st, et = int(row["start_time"]), int(row["end_time"])
            gs = row["gesture"]
            label = int(row["error1_nor0"])
            frame_features = get_frame_features_in_range(st, et, self.features_dict)
            if len(frame_features) == 0:
                continue
            gesture_id = self._parse_gesture_id(gs)
            for fn, feat in frame_features:
                self.features_seq.append(feat)
                labels.append(label)
                gestures.append(gesture_id)
                self._missing_kin_total += 1
                if fn in self.kine.index:
                    row_vals = self.kine.loc[fn].to_numpy(dtype=np.float32, copy=True)
                else:
                    self._missing_kin_count += 1
                    if not self._missing_kin_warned:
                        print(
                            f"[WARN] Missing kinematics index for some frames in {self.kinematics_path}. "
                            f"Falling back to zeros for missing indices."
                        )
                        self._missing_kin_warned = True
                    row_vals = np.zeros((self.kine_dim,), dtype=np.float32)
                kine_rows.append(row_vals)
        if len(kine_rows) > 0:
            self.kine_seq = torch.tensor(np.stack(kine_rows, axis=0), dtype=torch.float32)
        else:
            self.kine_seq = torch.empty((0, self.kine_dim), dtype=torch.float32)
        self.labels_seq = torch.tensor(labels, dtype=torch.long)
        self.gestures_seq = torch.tensor(gestures, dtype=torch.long)

    def __len__(self) -> int:
        return 1 if len(self.features_seq) > 0 else 0

    def __getitem__(self, index: int):
        if len(self.features_seq) == 0:
            return None
        return self.features_seq, self.gestures_seq, self.labels_seq, self.kine_seq


class JIGSAWS_Gesture_Kin_Features_SegmentWrapper(Dataset):
    """
    Segments the variable-length sequence into fixed-window segments (variable length allowed).
    """

    def __init__(self, original_dataset: JIGSAWS_Gesture_Kin_Features, segment_length: int = 40, step_size: int = 6):
        self.original_dataset = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.feature_dim = original_dataset.feature_dim
        self.kine_dim = original_dataset.kine_dim
        self.indices: list[tuple[int, int]] = []

        for idx in range(len(original_dataset)):
            data = original_dataset[idx]
            if data is None:
                continue
            features, _, _, kine = data
            total_frames = len(features)
            if total_frames <= 0:
                continue
            if total_frames - segment_length + 1 <= 0:
                self.indices.append((idx, 0))
            for start_frame in range(0, total_frames - segment_length + 1, step_size):
                self.indices.append((idx, start_frame))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        original_index, start_frame = self.indices[index]
        data = self.original_dataset[original_index]
        if data is None:
            return None
        features, gestures, labels, kine = data

        end_frame = min(start_frame + self.segment_length, len(features))
        seg_feats = features[start_frame:end_frame]
        seg_kine = kine[start_frame:end_frame, :]
        seg_labels = labels[start_frame:end_frame]
        seg_gestures = gestures[start_frame:end_frame]
        if len(seg_feats) == 0:
            return None

        seg_feat_tensor = torch.stack(seg_feats)  # (T, D)
        seg_kine_tensor = seg_kine.to(torch.float32)  # (T, K)
        return seg_feat_tensor, seg_kine_tensor, seg_labels, seg_gestures


def collate_fn_kin_features(batch):
    """
    Returns:
      features_padded: (B, T, D)
      kine_padded:     (B, T, K)
      masks:           (B, T) bool, True=valid
      labels:          (B,)
      gestures:        (B,)
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    features, kine, labels, gestures = zip(*batch)

    features = [f.clone().to(torch.float32) if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32) for f in features]
    kine = [k.clone().to(torch.float32) if isinstance(k, torch.Tensor) else torch.tensor(k, dtype=torch.float32) for k in kine]

    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    kine_padded = pad_sequence(kine, batch_first=True, padding_value=0.0)

    masks = torch.arange(features_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    gestures_padded = pad_sequence(gestures, batch_first=True, padding_value=0)
    return features_padded, kine_padded, masks, labels_padded, gestures_padded


class JIGSAWSDataKinFeatures:
    """
    Data loader class for pre-extracted visual features + kinematics.
    Mirrors `JIGSAWSDataFeatures` but adds kinematics.
    """

    def __init__(
        self,
        dataset_variant: str,
        model_name: str = "resnet50",
        data_root: str = "./data",
        split_root: Optional[str] = None,
        feature_root: str | None = None,
        kin_downsample: int = 3,
        frame_subsample: int = 1,
        debug_kin: bool = False,
    ):
        self.model_name = model_name
        self.data_root = data_root
        self.split_root = split_root
        self.feature_root = feature_root
        self.kin_downsample = kin_downsample
        self.frame_subsample = max(1, int(frame_subsample))
        self.debug_kin = debug_kin or (os.environ.get("KIN_DEBUG", "0") == "1")
        self.feature_dim = infer_feature_dim_from_root(feature_root) if feature_root else get_feature_dim(model_name)

        self.task, _, _ = parse_dataset_variant(dataset_variant)
        self.list_dataset_variant = build_dataset_variant_map(self.task)

        assert dataset_variant in self.list_dataset_variant.keys(), f"{dataset_variant} is not a valid dataset variant"

        data_split = self.split_selector(case=dataset_variant)
        self.train_records = data_split["train"]
        self.test_records = data_split["test"]

        self.build_train_dataset()
        self.build_test_dataset()

    def split_selector(self, case: str = "Suturing-l2"):
        return load_split_records(case, split_root=self.split_root)

    def _build_split(self, filenames):
        iterable_dataset = []
        for filename in filenames:
            base = JIGSAWS_Gesture_Kin_Features(
                filename,
                self.task,
                model_name=self.model_name,
                data_root=self.data_root,
                feature_root=self.feature_root,
                kin_downsample=self.kin_downsample,
                frame_subsample=self.frame_subsample,
                debug_kin=self.debug_kin,
            )
            wrapped = JIGSAWS_Gesture_Kin_Features_SegmentWrapper(base)
            iterable_dataset.append(wrapped)
        return iterable_dataset

    def build_train_dataset(self):
        self.train_dataset = self._build_split(self.train_records)

    def build_test_dataset(self):
        self.test_dataset = self._build_split(self.test_records)


