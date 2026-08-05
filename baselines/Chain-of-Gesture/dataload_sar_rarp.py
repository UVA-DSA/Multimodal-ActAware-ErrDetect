import csv
import os
import pickle
import re
from bisect import bisect_right
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


TRAIN_FEATURE_SPLIT = "training_set"
TEST_FEATURE_SPLIT = "testing_set"
TRAIN_GESTURE_SPLIT = "train"
TEST_GESTURE_SPLIT = "test"
TRAIN_PKL_DIR = "train_emb_DINOv2"
TEST_PKL_DIR = "test_emb_DINOv2"


def _video_sort_key(video_id: str) -> Tuple[int, ...]:
    parts = re.findall(r"\d+", video_id)
    return tuple(int(part) for part in parts) or (0,)


def _list_video_ids_from_pkl_dir(pkl_dir: str) -> List[str]:
    video_ids = []
    for file_name in os.listdir(pkl_dir):
        if file_name.endswith(".pkl"):
            video_ids.append(os.path.splitext(file_name)[0])
    return sorted(video_ids, key=_video_sort_key)


def list_video_ids(data_root: str, train: bool = True) -> List[str]:
    pkl_dir = os.path.join(data_root, TRAIN_PKL_DIR if train else TEST_PKL_DIR)
    return _list_video_ids_from_pkl_dir(pkl_dir)


def _get_split_names(train: bool) -> Tuple[str, str, str]:
    if train:
        return TRAIN_PKL_DIR, TRAIN_FEATURE_SPLIT, TRAIN_GESTURE_SPLIT
    return TEST_PKL_DIR, TEST_FEATURE_SPLIT, TEST_GESTURE_SPLIT


def _load_video_pickle(data_root: str, video_id: str, train: bool) -> dict:
    pkl_dir_name, _, _ = _get_split_names(train)
    video_path = os.path.join(data_root, pkl_dir_name, video_id + ".pkl")
    with open(video_path, "rb") as handle:
        return pickle.load(handle)


def _load_feature_array(data_root: str, video_id: str, train: bool, feature_source: str, video_data: dict) -> np.ndarray:
    if feature_source == "pkl":
        if "feature" not in video_data:
            raise KeyError("Missing `feature` key in {}".format(video_id))
        return np.asarray(video_data["feature"], dtype=np.float32)

    if feature_source == "npy":
        _, feature_split_name, _ = _get_split_names(train)
        feature_path = os.path.join(data_root, feature_split_name, video_id, "embed", "embed.npy")
        return np.load(feature_path).astype(np.float32)

    raise ValueError("Unsupported feature_source: {}".format(feature_source))


def _read_action_discrete_file(gesture_path: str) -> Tuple[np.ndarray, np.ndarray]:
    frame_numbers = []
    gesture_labels = []
    with open(gesture_path, "r") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            frame_numbers.append(int(row[0]))
            gesture_labels.append(int(row[1]))

    if not frame_numbers:
        raise ValueError("No gesture annotations found in {}".format(gesture_path))

    return np.asarray(frame_numbers, dtype=np.int64), np.asarray(gesture_labels, dtype=np.int64)


def _load_action_discrete_annotations(
    data_root: str,
    video_id: str,
    train: bool,
    max_frame_number: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    _, _, gesture_split_name = _get_split_names(train)
    gesture_root = os.path.join(
        data_root,
        "gestures",
        gesture_split_name,
    )
    gesture_path = os.path.join(gesture_root, video_id, "action_discrete.txt")
    if not os.path.exists(gesture_path):
        prefix = video_id + "_"
        candidate_names = sorted(
            [
                name
                for name in os.listdir(gesture_root)
                if name.startswith(prefix) and os.path.isdir(os.path.join(gesture_root, name))
            ],
            key=_video_sort_key,
        )
        if not candidate_names:
            raise FileNotFoundError("Missing gesture annotations for {}".format(video_id))

        chosen_path = None
        chosen_surplus = None
        longest_path = None
        longest_max_frame = -1
        for candidate_name in candidate_names:
            candidate_path = os.path.join(gesture_root, candidate_name, "action_discrete.txt")
            candidate_frames, _ = _read_action_discrete_file(candidate_path)
            candidate_max_frame = int(candidate_frames[-1])
            if candidate_max_frame > longest_max_frame:
                longest_max_frame = candidate_max_frame
                longest_path = candidate_path
            if max_frame_number is None or candidate_max_frame >= max_frame_number:
                surplus = candidate_max_frame - (max_frame_number or 0)
                if chosen_surplus is None or surplus < chosen_surplus:
                    chosen_surplus = surplus
                    chosen_path = candidate_path

        gesture_path = chosen_path or longest_path

    return _read_action_discrete_file(gesture_path)


def _frame_name_to_number(frame_name: str) -> int:
    stem = os.path.splitext(os.path.basename(frame_name))[0]
    return int(stem)


def _align_gesture_labels(frame_names: Sequence[str], gesture_frames: np.ndarray, gesture_labels: np.ndarray) -> np.ndarray:
    aligned = []
    gesture_frames_list = gesture_frames.tolist()
    gesture_label_lookup = {
        int(frame_number): int(label)
        for frame_number, label in zip(gesture_frames, gesture_labels)
    }

    for frame_name in frame_names:
        frame_number = _frame_name_to_number(str(frame_name))
        if frame_number in gesture_label_lookup:
            aligned.append(gesture_label_lookup[frame_number])
            continue

        insert_idx = bisect_right(gesture_frames_list, frame_number) - 1
        if insert_idx < 0:
            insert_idx = 0
        aligned.append(int(gesture_labels[insert_idx]))

    return np.asarray(aligned, dtype=np.float32)


class SarRarpVideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        feature_source: str = "npy",
        train: bool = True,
        video_ids: Optional[Sequence[str]] = None,
        transform=None,
        mstcn: bool = False,
    ):
        self.data_root = data_root
        self.feature_source = feature_source
        self.train = train
        self.transform = transform
        self.mstcn = mstcn
        if video_ids is None:
            self.video_ids = list_video_ids(data_root, train=train)
        else:
            self.video_ids = sorted(list(video_ids), key=_video_sort_key)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int):
        video_id = self.video_ids[idx]
        video_data = _load_video_pickle(self.data_root, video_id, self.train)
        features = _load_feature_array(
            self.data_root,
            video_id,
            self.train,
            self.feature_source,
            video_data,
        )
        error_labels = np.asarray(video_data["error_GT"], dtype=np.float32)
        frame_names = np.asarray(video_data["image_name"])

        if len(frame_names) == 0:
            raise ValueError("No frame names found for {}".format(video_id))
        max_frame_number = _frame_name_to_number(str(frame_names[-1]))
        gesture_frames, gesture_labels = _load_action_discrete_annotations(
            self.data_root,
            video_id,
            self.train,
            max_frame_number=max_frame_number,
        )
        aligned_gesture_labels = _align_gesture_labels(frame_names, gesture_frames, gesture_labels)

        num_frames = min(len(features), len(error_labels), len(aligned_gesture_labels))
        features = features[:num_frames].astype(np.float32)
        error_labels = error_labels[:num_frames].astype(np.float32)
        aligned_gesture_labels = aligned_gesture_labels[:num_frames].astype(np.float32)

        if self.transform is not None:
            features = self.transform(features)

        return features, num_frames, error_labels, aligned_gesture_labels, video_id
