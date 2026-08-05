import os
from torch.utils.data import Dataset
import pickle
import numpy as np
import re
from typing import Optional, Tuple, List

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pandas as pd
import glob
from torch.nn.utils.rnn import pad_sequence

print("dataload_rarp_savedres.py from SAR_RARP loaded")

def extract_number(file_name: str) -> int:
    match = re.search(r"(\d+)", file_name)
    return int(match.group()) if match else 0
class CustomVideoDatasetEmbedTri(Dataset):
    def __init__(self, root_dir: str, transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = True if "train" in self.root_dir else False
        self.video_folders = sorted(os.listdir(root_dir), key=extract_number)
        self.image_path = (
            "./data/SAR_RARP50/training_set" if self.train else "./data/SAR_RARP50/testing_set"
        )

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray, str]:
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name)
        image_path = os.path.join(self.image_path, video_name.replace(".pkl", ""),"embed","embed.npy")
        ges_path = os.path.join(self.image_path, video_name.replace(".pkl", ""),"embed","tri_embed.npy")
        with open(video_path, "rb") as f:
            video_data = pickle.load(f)
        embeddings = np.load(image_path)
        embeddings_ges = np.load(ges_path)
        features = embeddings.astype("float32")
        ges_features = embeddings_ges.astype("float32")
        e_labels = video_data["error_GT"]
        video_length = len(e_labels)

        return features, ges_features, video_length, e_labels, video_name

class CustomVideoDatasetEmbed(Dataset):
    def __init__(self, root_dir: str, transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = True if "train" in self.root_dir else False
        self.video_folders = sorted(os.listdir(root_dir), key=extract_number)
        self.image_path = (
            "./data/SAR_RARP50/training_set" if self.train else "./data/SAR_RARP50/testing_set"
        )

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray, str]:
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name)
        image_path = os.path.join(self.image_path, video_name.replace(".pkl", ""),"embed","embed.npy") # saved resnet embeddings gesture and triplet
        ges_path = os.path.join(self.image_path, video_name.replace(".pkl", ""),"embed","ges_embed.npy")
        with open(video_path, "rb") as f:
            video_data = pickle.load(f)
        embeddings = np.load(image_path)
        embeddings_ges = np.load(ges_path)
        features = embeddings.astype("float32")
        ges_features = embeddings_ges.astype("float32")
        e_labels = video_data["error_GT"]
        video_length = len(e_labels)

        return features, ges_features, video_length, e_labels, video_name
    
class Gesture_SegmentWrapperEmbed(Dataset):
    def __init__(self, original_dataset, segment_length=20, step_size=20):
        self.original_dataset = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.indices = []

        # Build indices to allow indexing by segment
        for idx in range(len(original_dataset)):
            features, ges_features, video_length, e_labels, video_name = self.original_dataset[idx]
            total_frames = len(features)
            if total_frames == 0:
                continue
            # Generate start indices for each segment
            for start_frame in range(0, total_frames - segment_length + 1, step_size):
                self.indices.append((idx, start_frame))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index, start_frame = self.indices[index]
        features,ges_features, video_length, e_labels, video_name = self.original_dataset[original_index]

        end_frame = start_frame + self.segment_length
        segment_features = features[start_frame:end_frame]
        segment_gesfeatures = ges_features[start_frame:end_frame]
        segment_labels = e_labels[start_frame:end_frame]
        max_label = np.max(segment_labels)
        segment_labels = np.full_like(segment_labels, max_label)
        return segment_features, segment_gesfeatures, segment_labels, video_name

  

class CustomVideoDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = True if "train" in self.root_dir else False
        self.video_folders = sorted(os.listdir(root_dir), key=extract_number)
        self.image_path = (
            "./data/SAR_RARP50/training_set" if self.train else "./data/SAR_RARP50/testing_set"
        )

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray, str]:
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name)
        image_path = os.path.join(self.image_path, video_name.replace(".pkl", ""),"embed","embed.npy")
        with open(video_path, "rb") as f:
            video_data = pickle.load(f)
        embeddings = np.load(image_path)
        features = embeddings.astype("float32")
        e_labels = video_data["error_GT"]
        video_length = len(e_labels)

        return features, video_length, e_labels, video_name

class Gesture_SegmentWrapper(Dataset):
    def __init__(self, original_dataset, segment_length=40, step_size=6):
        self.original_dataset = original_dataset
        self.segment_length = segment_length
        self.step_size = step_size
        self.indices = []

        # Build indices to allow indexing by segment
        for idx in range(len(original_dataset)):
            features, video_length, e_labels, video_name = self.original_dataset[idx]
            total_frames = len(features)
            if total_frames == 0:
                continue
            # Generate start indices for each segment
            for start_frame in range(0, total_frames - segment_length + 1, step_size):
                self.indices.append((idx, start_frame))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index, start_frame = self.indices[index]
        features, video_length, e_labels, video_name = self.original_dataset[original_index]

        end_frame = start_frame + self.segment_length
        segment_features = features[start_frame:end_frame]
        segment_labels = e_labels[start_frame:end_frame]
        max_label = np.max(segment_labels)
        # Replace the segment's labels with the max value, preserving the length
        segment_labels = np.full_like(segment_labels, max_label)

        # Return the segmented features, their labels, and the video name
        return segment_features, segment_labels, video_name

  