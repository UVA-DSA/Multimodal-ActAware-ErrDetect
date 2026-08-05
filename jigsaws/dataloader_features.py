"""
Feature-based Dataloader for JIGSAWS Dataset

This module provides dataset classes that load pre-extracted visual features
instead of raw images, avoiding redundant CNN/ViT computation during training.

Supports features extracted from both ResNet and CLIP vision encoders.

Use this dataloader after running extract_features.py to pre-compute features.
"""

import os
import torch
import re
import random
import glob
from typing import Optional
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from jigsaws_splits import build_dataset_variant_map, load_split_records, parse_dataset_variant


# Feature dimensions for each model (must match feature extraction scripts)
MODEL_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "clip-vit-base-patch32": 768,
    "clip-vit-base-patch16": 768,
    "clip-vit-large-patch14": 1024,
    "clip-vit-large-patch14-336": 1024,
    "surgvlp": 768,
}


def get_feature_dim(model_name):
    """Get the feature dimension for a given model name."""
    if model_name not in MODEL_FEATURE_DIMS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FEATURE_DIMS.keys())}")
    return MODEL_FEATURE_DIMS[model_name]


def default_feature_root(data_root: str, model_name: str) -> str:
    return os.path.join(data_root, "vid_features", model_name)


def resolve_feature_path(
    task: str,
    video_name: str,
    *,
    data_root: str = "./data",
    model_name: str = "resnet50",
    feature_root: Optional[str] = None,
) -> str:
    root = feature_root or default_feature_root(data_root, model_name)
    return os.path.join(root, task, video_name, "features.pt")


def infer_feature_dim_from_root(feature_root: str, tasks: tuple[str, ...] = ("Suturing", "Needle_Passing")) -> int:
    checked_dirs = []
    for task in tasks:
        task_dir = os.path.join(feature_root, task)
        checked_dirs.append(task_dir)
        if not os.path.isdir(task_dir):
            continue
        for video_name in sorted(os.listdir(task_dir)):
            fp = os.path.join(task_dir, video_name, "features.pt")
            if not os.path.exists(fp):
                continue
            d = torch.load(fp, map_location="cpu")
            if len(d) == 0:
                continue
            return int(next(iter(d.values())).shape[0])
    raise FileNotFoundError(
        f"Could not infer feature dim from {feature_root}. "
        f"Checked task dirs: {checked_dirs}"
    )


def get_frame_features_in_range(start_time, end_time, features_dict):
    """
    Get features for frames within a given time range.
    
    Args:
        start_time: Start frame number in the original annotation space
        end_time: End frame number in the original annotation space
        features_dict: Dict mapping saved frame_number -> feature tensor
    
    Returns:
        List of (frame_number, feature) tuples sorted by frame number

    Note:
        JIGSAWS frame PNGs are already downsampled (for example `0, 3, 6, ...`)
        but keep their original frame numbers. The interval filtering here
        intentionally uses those preserved numbers directly.
    """
    frames_in_range = []
    for frame_num, feature in features_dict.items():
        if start_time <= frame_num <= end_time:
            frames_in_range.append((frame_num, feature))
    
    # Sort by frame number
    frames_in_range.sort(key=lambda x: x[0])
    return frames_in_range


class JIGSAWS_Gesture_Features(Dataset):
    """
    Dataset that loads pre-extracted features instead of raw images.
    
    Args:
        filename: CSV filename (e.g., "Suturing_S02_T01.csv")
        task: Task name (e.g., "Suturing", "Needle_Passing")
        model_name: Name of the feature extractor model (e.g., "resnet50", "clip-vit-base-patch32")
        data_root: Root data directory (default: "./data")
        frame_subsample: Keep every k-th stored frame. Stored JIGSAWS frames are at
            10 Hz (original frame ids 0, 3, 6, ...); use 2 for the paper's 5 Hz rate.
    """
    def __init__(self, filename, task, model_name="resnet50", data_root="./data", feature_root=None,
                 frame_subsample=1):
        self.error_dirs = f"{data_root}/{task}/errors/{filename}"
        self.curtb = pd.read_csv(self.error_dirs)
        self.label = self.curtb['error1_nor0']
        self.filename = filename
        self.task = task
        self.model_name = model_name
        self.data_root = data_root
        self.feature_root = feature_root
        self.feature_dim = MODEL_FEATURE_DIMS.get(model_name)
        if self.feature_dim is None and feature_root is None:
            self.feature_dim = get_feature_dim(model_name)
        
        # Load pre-extracted features from model-specific directory
        video_name = filename[:-4]  # Remove .csv extension
        self.video_name = video_name
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
            elif model_name == "surgvlp":
                print("[WARN] Run: python extract_surgvlp_features_jigsaws.py")
            else:
                print(f"[WARN] Run: python extract_features.py --model {model_name}")
            self.features_dict = {}
        else:
            self.features_dict = torch.load(self.features_path, map_location="cpu")
            # Verify feature dimension
            if len(self.features_dict) > 0:
                sample_feat = next(iter(self.features_dict.values()))
                actual_dim = sample_feat.shape[0]
                if self.feature_dim is not None and actual_dim != self.feature_dim:
                    print(f"[WARN] Feature dim mismatch: expected {self.feature_dim}, got {actual_dim}")
                self.feature_dim = actual_dim
            print(f"[INFO] Loaded {len(self.features_dict)} frame features (dim={self.feature_dim}) from {self.features_path}")
        self.frame_subsample = max(1, int(frame_subsample))
        if self.frame_subsample > 1 and len(self.features_dict) > 0:
            kept = sorted(self.features_dict.keys())[:: self.frame_subsample]
            self.features_dict = {k: self.features_dict[k] for k in kept}
        self._build_sequence()

    @staticmethod
    def _parse_gesture_id(gs):
        gs_str = str(gs)
        if gs_str.startswith("G"):
            return int(gs_str[1:])
        return int(gs_str)

    def _build_sequence(self):
        self.features_seq = []
        labels = []
        gestures = []
        frame_nums = []
        for _, row in self.curtb.iterrows():
            st, et = int(row["start_time"]), int(row["end_time"])
            gs = row["gesture"]
            label = int(row["error1_nor0"])
            frame_features = get_frame_features_in_range(st, et, self.features_dict)
            if len(frame_features) == 0:
                continue
            gesture_id = self._parse_gesture_id(gs)
            for frame_num, feat in frame_features:
                self.features_seq.append(feat)
                labels.append(label)
                gestures.append(gesture_id)
                frame_nums.append(int(frame_num))
        self.labels_seq = torch.tensor(labels, dtype=torch.long)
        self.gestures_seq = torch.tensor(gestures, dtype=torch.long)
        self.frame_nums_seq = torch.tensor(frame_nums, dtype=torch.long)
    
    def __getitem__(self, index):
        if len(self.features_seq) == 0:
            return None
        return self.features_seq, self.gestures_seq, self.labels_seq
    
    def __len__(self):
        return 1 if len(self.features_seq) > 0 else 0


class JIGSAWS_Gesture_Features_SegmentWrapper(Dataset):
    """
    Wrapper that segments feature sequences into fixed-length segments.
    """
    def __init__(self, original_dataset, segment_length=40, step_size=6):
        self.original_dataset = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.feature_dim = original_dataset.feature_dim
        self.indices = []
        
        for idx in range(len(original_dataset)):
            data = original_dataset[idx]
            if data is None:
                continue
            features, _, _ = data
            total_frames = len(features)
            
            # Only consider samples that have at least 1 frame
            if total_frames <= 0:
                # print(f"[WARN] Skipping dataset index {idx} due to zero frames: file={self.original_dataset.filename}")
                continue
            
            # If fewer than a segment, still take the first partial segment starting at 0
            if total_frames - segment_length + 1 <= 0:
                self.indices.append((idx, 0))
            
            for start_frame in range(0, total_frames - segment_length + 1, step_size):
                self.indices.append((idx, start_frame))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        original_index, start_frame = self.indices[index]
        data = self.original_dataset[original_index]
        if data is None:
            return None
        features, gestures, labels = data
        
        end_frame = min(start_frame + self.segment_length, len(features))
        segment_features = features[start_frame:end_frame]
        segment_labels = labels[start_frame:end_frame]
        segment_gestures = gestures[start_frame:end_frame]
        
        if len(segment_features) == 0:
            # print(f"[WARN] Empty segment: idx={original_index}, start={start_frame}, file={self.original_dataset.filename}")
            return None
        
        # Stack features into a single tensor
        # Shape: (segment_length, feature_dim)
        segment_tensor = torch.stack(segment_features)
        
        return segment_tensor, segment_labels, segment_gestures


class JIGSAWS_Gesture_Features_SegmentWrapperWithMeta(Dataset):
    """
    Segment wrapper that also returns frame numbers and video/task metadata.
    """
    def __init__(self, original_dataset, segment_length=40, step_size=6):
        self.original_dataset = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.feature_dim = original_dataset.feature_dim
        self.indices = []

        for idx in range(len(original_dataset)):
            data = original_dataset[idx]
            if data is None:
                continue
            features, _, _ = data
            total_frames = len(features)
            if total_frames <= 0:
                continue
            if total_frames - segment_length + 1 <= 0:
                self.indices.append((idx, 0))
            for start_frame in range(0, total_frames - segment_length + 1, step_size):
                self.indices.append((idx, start_frame))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index, start_frame = self.indices[index]
        data = self.original_dataset[original_index]
        if data is None:
            return None
        features, gestures, labels = data
        frame_nums = self.original_dataset.frame_nums_seq

        end_frame = min(start_frame + self.segment_length, len(features))
        segment_features = features[start_frame:end_frame]
        segment_labels = labels[start_frame:end_frame]
        segment_gestures = gestures[start_frame:end_frame]
        segment_frame_nums = frame_nums[start_frame:end_frame]

        if len(segment_features) == 0:
            return None

        segment_tensor = torch.stack(segment_features)
        return (
            segment_tensor,
            segment_labels,
            segment_gestures,
            segment_frame_nums,
            self.original_dataset.video_name,
            self.original_dataset.task,
        )


def collate_fn_features(batch):
    """
    Collate function for feature-based batches (segment-level prediction).
    
    Returns one label per segment (not repeated across frames).
    Returns: (features_padded, masks, labels, gestures)
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    
    features, labels, gestures = zip(*batch)
    
    # Convert to tensors if needed
    features = [f.clone().to(torch.float32) if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32) 
                for f in features]
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    
    # Pad the sequences along the 0th dimension (sequence length)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    
    # Create masks for the padded sequences (True = valid, False = padding)
    masks = torch.arange(features_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    gestures_padded = pad_sequence(gestures, batch_first=True, padding_value=0)
    
    return features_padded, masks, labels_padded, gestures_padded


def collate_fn_features_context(batch):
    """
    Collate function for context-based models (per-frame prediction).
    
    Returns labels repeated across frames with masks for padding.
    Returns: (features_padded, masks, labels_padded, gestures_padded)
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    
    features, labels, gestures = zip(*batch)
    
    # Convert to tensors if needed
    features = [f.clone().to(torch.float32) if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32) 
                for f in features]
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    
    # Pad the sequences along the 0th dimension (sequence length)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    
    # Create masks for the padded sequences
    masks = torch.arange(features_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    gestures_padded = pad_sequence(gestures, batch_first=True, padding_value=0)
    
    return features_padded, masks, labels_padded, gestures_padded


def collate_fn_features_with_meta(batch):
    """
    Collate function that includes frame numbers and video/task metadata.
    Returns: (features_padded, masks, labels_padded, gestures_padded, frame_nums_padded, video_names, tasks)
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    features, labels, gestures, frame_nums, video_names, tasks = zip(*batch)
    features = [f.clone().to(torch.float32) if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
                for f in features]
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    masks = torch.arange(features_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    gestures_padded = pad_sequence(gestures, batch_first=True, padding_value=0)
    frame_nums_padded = pad_sequence(frame_nums, batch_first=True, padding_value=-1)

    return features_padded, masks, labels_padded, gestures_padded, frame_nums_padded, list(video_names), list(tasks)


class JIGSAWSDataFeatures():
    """
    Data loader class for pre-extracted features.
    
    Args:
        dataset_variant: Dataset variant string (e.g., "Suturing-ls1")
        model_name: Name of the feature extractor model (e.g., "resnet50", "clip-vit-base-patch32")
        data_root: Root data directory (default: "./data")
    """
    def __init__(self, dataset_variant, model_name="resnet50", data_root="./data", split_root=None, feature_root=None,
                 frame_subsample=1):
        self.model_name = model_name
        self.data_root = data_root
        self.split_root = split_root
        self.feature_root = feature_root
        self.frame_subsample = max(1, int(frame_subsample))
        self.feature_dim = infer_feature_dim_from_root(feature_root) if feature_root else get_feature_dim(model_name)

        self.task, _, _ = parse_dataset_variant(dataset_variant)
        self.list_dataset_variant = build_dataset_variant_map(self.task)

        assert dataset_variant in self.list_dataset_variant.keys(), f"{dataset_variant} is not a valid dataset variant"

        data_split = self.split_selector(case=dataset_variant)

        self.train_records = data_split['train']
        self.test_records = data_split['test']

        self.build_train_dataset()
        self.build_test_dataset()
    
    def split_selector(self, case='Suturing-l2'):
        return load_split_records(case, split_root=self.split_root)
    
    def build_train_dataset(self):
        iterable_dataset = []
        for filename in self.train_records:
            dataset = JIGSAWS_Gesture_Features(
                filename,
                self.task,
                self.model_name,
                self.data_root,
                feature_root=self.feature_root,
                frame_subsample=self.frame_subsample,
            )
            dataset = JIGSAWS_Gesture_Features_SegmentWrapper(dataset)
            iterable_dataset.append(dataset)
        self.train_dataset = iterable_dataset

    def build_test_dataset(self):
        iterable_dataset = []
        for filename in self.test_records:
            dataset = JIGSAWS_Gesture_Features(
                filename,
                self.task,
                self.model_name,
                self.data_root,
                feature_root=self.feature_root,
                frame_subsample=self.frame_subsample,
            )
            dataset = JIGSAWS_Gesture_Features_SegmentWrapper(dataset)
            iterable_dataset.append(dataset)
        self.test_dataset = iterable_dataset
    
    def build(self):
        return (self.train_dataset, self.test_dataset)
