"""
SAR-RARP50 feature dataloaders.

Supports:
- Base features:
  - Existing ResNet50 features saved as `embed.npy` under:
      data/SAR_RARP50/{training_set,testing_set}/{video}/embed/embed.npy
  - Newly extracted features (e.g. CLIP ViT or a SAR-RARP50 error-finetuned ResNet50)
    saved as `features.pt` dicts under:
      data/SAR_RARP50/vid_features/{model}/{training_set,testing_set}/{video}/features.pt

- Gesture/context branch features for cross-context training:
  - `ges_embed.npy` or `tri_embed.npy` under the same `embed/` folder.

Labels:
- Frame-level labels are loaded from the `.pkl` files under:
    data/SAR_RARP50/{train_emb_DINOv2,test_emb_DINOv2}/*.pkl
  We use `error_GT` and convert each sliding-window segment to a single scalar label
  (max over frames) for segment-level training, or repeat that scalar per-frame for
  per-frame context training.
"""

from __future__ import annotations

import os
import pickle
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _extract_number(s: str) -> int:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _load_error_labels(pkl_path: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        video_data = pickle.load(f)
    if "error_GT" not in video_data:
        raise KeyError(f"Missing `error_GT` in {pkl_path}")
    return np.asarray(video_data["error_GT"], dtype=np.int64)


def _video_id_from_pkl(pkl_filename: str) -> str:
    # "video_01.pkl" -> "video_01"
    return os.path.basename(pkl_filename).replace(".pkl", "")


def _load_features_embed(data_root: str, split_set: str, video_id: str) -> np.ndarray:
    # data_root points to .../data/SAR_RARP50
    fp = os.path.join(data_root, split_set, video_id, "embed", "embed.npy")
    return np.load(fp).astype("float32")


def _load_features_gesture_embed(data_root: str, split_set: str, video_id: str, kind: str) -> np.ndarray:
    # kind: "ges" -> ges_embed.npy, "tri" -> tri_embed.npy
    if kind not in {"ges", "tri"}:
        raise ValueError(f"Unknown gesture embed kind: {kind}")
    name = "ges_embed.npy" if kind == "ges" else "tri_embed.npy"
    fp = os.path.join(data_root, split_set, video_id, "embed", name)
    return np.load(fp).astype("float32")


def _load_features_pt(data_root: str, model_name: str, split_set: str, video_id: str) -> np.ndarray:
    # Stored as dict[int -> tensor(D)] like root `extract_features.py`.
    fp = os.path.join(data_root, "vid_features", model_name, split_set, video_id, "features.pt")
    d: Dict[int, torch.Tensor] = torch.load(fp, map_location="cpu")
    if len(d) == 0:
        return np.zeros((0, 1), dtype="float32")
    keys = sorted(d.keys())
    feats = torch.stack([d[k].to(torch.float32) for k in keys], dim=0)
    return feats.numpy()


class _LRUCache:
    """
    Tiny per-worker LRU cache (Dataset objects are replicated per worker).
    """

    def __init__(self, max_items: int = 16):
        self.max_items = int(max_items)
        self._d: "OrderedDict[str, object]" = OrderedDict()

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


@dataclass(frozen=True)
class BaseFeatureSpec:
    source: str  # "embed" or "pt"
    model: Optional[str] = None  # required when source=="pt"


def split_train_val(items: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    items = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    val = items[:n_val]
    train = items[n_val:]
    return train, val


class SAR_RARP50SegmentDataset(Dataset):
    """
    Segment-level dataset.

    Returns: (features[T,D], label_scalar, video_id)
    """

    def __init__(
        self,
        *,
        data_root: str,
        pkl_dir: str,
        split_set: str,
        base: BaseFeatureSpec,
        video_ids: Optional[Sequence[str]] = None,
        segment_length: int = 40,
        step_size: int = 6,
        cache_max_videos: int = 8,
        cache_all_videos: bool = True,
        label_mode: str = "segment",  # "segment" or "frame"
    ):
        self.data_root = data_root
        self.pkl_dir = pkl_dir
        self.split_set = split_set
        self.base = base
        self.segment_length = int(segment_length)
        self.step_size = int(step_size)
        self.label_mode = str(label_mode)
        self.cache_all_videos = bool(cache_all_videos)
        if self.label_mode not in {"segment", "frame"}:
            raise ValueError(f"label_mode must be one of {{'segment','frame'}}, got {self.label_mode}")

        # Per-worker caches (video_id -> tensor)
        self._cache_labels = _LRUCache(max_items=cache_max_videos)
        self._cache_base = _LRUCache(max_items=cache_max_videos)
        self._labels_all: Dict[str, torch.Tensor] = {}
        self._base_all: Dict[str, torch.Tensor] = {}

        all_pkls = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")], key=_extract_number)
        all_video_ids = [_video_id_from_pkl(p) for p in all_pkls]
        if video_ids is None:
            self.video_ids = all_video_ids
        else:
            want = set(video_ids)
            self.video_ids = [v for v in all_video_ids if v in want]

        self._pkl_by_video = {vid: os.path.join(pkl_dir, f"{vid}.pkl") for vid in self.video_ids}
        print(self.split_set, self.video_ids)

        if self.cache_all_videos:
            for vid in self.video_ids:
                self._labels_all[vid] = torch.from_numpy(
                    _load_error_labels(self._pkl_by_video[vid])
                ).to(torch.long)
                self._base_all[vid] = self._load_base(vid)

        # Precompute segment indices (video_id, start_frame)
        self.indices: List[Tuple[str, int]] = []
        for vid in self.video_ids:
            labels = self._get_labels_tensor(vid)
            n = int(labels.numel())
            if n <= 0:
                continue
            if n - self.segment_length + 1 <= 0:
                self.indices.append((vid, 0))
            else:
                for s in range(0, n - self.segment_length + 1, self.step_size):
                    self.indices.append((vid, s))

    def __len__(self) -> int:
        return len(self.indices)

    def _load_base(self, vid: str) -> np.ndarray:
        if self.base.source == "embed":
            arr = _load_features_embed(self.data_root, self.split_set, vid)
            return torch.from_numpy(arr).to(torch.float32)
        if self.base.source == "pt":
            if not self.base.model:
                raise ValueError("BaseFeatureSpec.model is required when source=='pt'")
            arr = _load_features_pt(self.data_root, self.base.model, self.split_set, vid)
            return torch.from_numpy(arr).to(torch.float32)
        raise ValueError(f"Unknown base feature source: {self.base.source}")

    def _get_labels_tensor(self, vid: str) -> torch.Tensor:
        if self.cache_all_videos and vid in self._labels_all:
            return self._labels_all[vid]
        key = f"lab::{vid}"
        cached = self._cache_labels.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        labels = _load_error_labels(self._pkl_by_video[vid])
        t = torch.from_numpy(labels).to(torch.long)
        self._cache_labels.put(key, t)
        return t

    def _get_base_tensor(self, vid: str) -> torch.Tensor:
        if self.cache_all_videos and vid in self._base_all:
            return self._base_all[vid]
        key = f"base::{self.base.source}::{self.base.model or 'none'}::{self.split_set}::{vid}"
        cached = self._cache_base.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        feats = self._load_base(vid)
        self._cache_base.put(key, feats)
        return feats

    def __getitem__(self, idx: int):
        vid, start = self.indices[idx]
        labels = self._get_labels_tensor(vid)  # (T,)
        feats = self._get_base_tensor(vid)     # (T,D)

        n_lab = int(labels.numel())
        n_feat = int(feats.size(0))
        n = min(n_lab, n_feat) if n_feat > 0 else n_lab
        if n <= 0:
            return None

        end = min(start + self.segment_length, n)
        seg_feats = feats[start:end]     # torch (t,D)
        seg_labels = labels[start:end]   # torch (t,)
        if seg_feats.size(0) == 0:
            return None
        if self.label_mode == "frame":
            # return per-frame labels for this segment window
            return seg_feats, seg_labels.to(torch.long), vid
        label_scalar = int(seg_labels.max().item()) if seg_labels.numel() else 0
        return seg_feats, torch.tensor(label_scalar, dtype=torch.long), vid


def collate_fn_segment(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    feats, labels, vids = zip(*batch)
    feats = [f.to(torch.float32) for f in feats]
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    masks = torch.arange(feats_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    labels_tensor = torch.tensor([int(l.item()) for l in labels], dtype=torch.long)
    return feats_padded, masks.to(torch.bool), labels_tensor, list(vids)

def collate_fn_segment_frame(batch):
    """
    Collate for frame-level labels: labels are sequences (T,) aligned to features.
    Returns:
      feats_padded (B,T,D), masks (B,T) bool, labels_padded (B,T) long, vids (list[str])
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    feats, labels, vids = zip(*batch)
    feats = [f.to(torch.float32) for f in feats]
    labels = [l.to(torch.long) for l in labels]
    lengths = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    masks = torch.arange(feats_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return feats_padded, masks.to(torch.bool), labels_padded, list(vids)


class SAR_RARP50DualContextDataset(Dataset):
    """
    Dual-feature dataset for per-frame context training.

    Returns: (base_feats[T,Db], ges_feats[T,Dg], labels_vec[T], video_id)
    where labels_vec is the segment-level scalar label repeated across frames.
    """

    def __init__(
        self,
        *,
        data_root: str,
        pkl_dir: str,
        split_set: str,
        base: BaseFeatureSpec,
        gesture_embed: str = "ges",  # "ges" or "tri"
        video_ids: Optional[Sequence[str]] = None,
        segment_length: int = 40,
        step_size: int = 6,
        cache_max_videos: int = 8,
        cache_all_videos: bool = True,
        label_mode: str = "segment",  # "segment" or "frame"
    ):
        self.data_root = data_root
        self.pkl_dir = pkl_dir
        self.split_set = split_set
        self.base = base
        self.gesture_embed = gesture_embed
        self.segment_length = int(segment_length)
        self.step_size = int(step_size)
        self.label_mode = str(label_mode)
        self.cache_all_videos = bool(cache_all_videos)
        if self.label_mode not in {"segment", "frame"}:
            raise ValueError(f"label_mode must be one of {{'segment','frame'}}, got {self.label_mode}")

        self._cache_labels = _LRUCache(max_items=cache_max_videos)
        self._cache_base = _LRUCache(max_items=cache_max_videos)
        self._cache_ges = _LRUCache(max_items=cache_max_videos * 2)
        self._labels_all: Dict[str, torch.Tensor] = {}
        self._base_all: Dict[str, torch.Tensor] = {}
        self._ges_all: Dict[str, torch.Tensor] = {}

        all_pkls = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")], key=_extract_number)
        all_video_ids = [_video_id_from_pkl(p) for p in all_pkls]
        if video_ids is None:
            self.video_ids = all_video_ids
        else:
            want = set(video_ids)
            self.video_ids = [v for v in all_video_ids if v in want]

        self._pkl_by_video = {vid: os.path.join(pkl_dir, f"{vid}.pkl") for vid in self.video_ids}

        if self.cache_all_videos:
            for vid in self.video_ids:
                self._labels_all[vid] = torch.from_numpy(
                    _load_error_labels(self._pkl_by_video[vid])
                ).to(torch.long)
                self._base_all[vid] = self._load_base(vid)
                self._ges_all[vid] = self._get_ges_tensor(vid)

        self.indices: List[Tuple[str, int]] = []
        for vid in self.video_ids:
            labels = self._get_labels_tensor(vid)
            n = int(labels.numel())
            if n <= 0:
                continue
            if n - self.segment_length + 1 <= 0:
                self.indices.append((vid, 0))
            else:
                for s in range(0, n - self.segment_length + 1, self.step_size):
                    self.indices.append((vid, s))

    def __len__(self) -> int:
        return len(self.indices)

    def _load_base(self, vid: str) -> np.ndarray:
        if self.base.source == "embed":
            arr = _load_features_embed(self.data_root, self.split_set, vid)
            return torch.from_numpy(arr).to(torch.float32)
        if self.base.source == "pt":
            if not self.base.model:
                raise ValueError("BaseFeatureSpec.model is required when source=='pt'")
            arr = _load_features_pt(self.data_root, self.base.model, self.split_set, vid)
            return torch.from_numpy(arr).to(torch.float32)
        raise ValueError(f"Unknown base feature source: {self.base.source}")

    def _get_labels_tensor(self, vid: str) -> torch.Tensor:
        if self.cache_all_videos and vid in self._labels_all:
            return self._labels_all[vid]
        key = f"lab::{vid}"
        cached = self._cache_labels.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        labels = _load_error_labels(self._pkl_by_video[vid])
        t = torch.from_numpy(labels).to(torch.long)
        self._cache_labels.put(key, t)
        return t

    def _get_base_tensor(self, vid: str) -> torch.Tensor:
        if self.cache_all_videos and vid in self._base_all:
            return self._base_all[vid]
        key = f"base::{self.base.source}::{self.base.model or 'none'}::{self.split_set}::{vid}"
        cached = self._cache_base.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        feats = self._load_base(vid)
        self._cache_base.put(key, feats)
        return feats

    def _get_ges_tensor(self, vid: str) -> torch.Tensor:
        if self.cache_all_videos and vid in self._ges_all:
            return self._ges_all[vid]
        key = f"ges::{self.gesture_embed}::{self.split_set}::{vid}"
        cached = self._cache_ges.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        arr = _load_features_gesture_embed(self.data_root, self.split_set, vid, self.gesture_embed)
        t = torch.from_numpy(arr).to(torch.float32)
        self._cache_ges.put(key, t)
        return t

    def __getitem__(self, idx: int):
        vid, start = self.indices[idx]
        labels = self._get_labels_tensor(vid)
        base = self._get_base_tensor(vid)
        ges = self._get_ges_tensor(vid)

        n = min(int(labels.numel()), int(base.size(0)), int(ges.size(0)))
        if n <= 0:
            return None

        end = min(start + self.segment_length, n)
        base_seg = base[start:end]
        ges_seg = ges[start:end]
        seg_labels = labels[start:end]
        if base_seg.size(0) == 0:
            return None

        if self.label_mode == "frame":
            labels_vec = seg_labels.to(torch.long)
        else:
            label_scalar = int(seg_labels.max().item()) if seg_labels.numel() else 0
            labels_vec = torch.full((base_seg.size(0),), label_scalar, dtype=torch.long)
        return (base_seg, ges_seg, labels_vec, vid)


def collate_fn_dual_context(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    base, ges, labels, vids = zip(*batch)
    base = [x.to(torch.float32) for x in base]
    ges = [x.to(torch.float32) for x in ges]
    labels = [y.to(torch.long) for y in labels]

    lengths = torch.tensor([x.size(0) for x in base], dtype=torch.long)
    base_padded = pad_sequence(base, batch_first=True, padding_value=0.0)
    ges_padded = pad_sequence(ges, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    masks = torch.arange(base_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return base_padded, ges_padded, masks.to(torch.bool), labels_padded, list(vids)


