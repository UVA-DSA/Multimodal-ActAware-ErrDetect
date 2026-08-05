import os
import pickle
import re
from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


def extract_number(file_name: str) -> int:
    match = re.search(r"(\d+)", file_name)
    return int(match.group()) if match else 0


def normalize_video_name(video_name: str) -> str:
    base_name = os.path.basename(str(video_name).strip())
    stem, ext = os.path.splitext(base_name)
    if ext == ".csv":
        return stem + ".pkl"
    if ext == ".pkl":
        return base_name
    return base_name + ".pkl"


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load PKLs written with NumPy 2.x under NumPy 1.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle_compat(path: str):
    with open(path, "rb") as handle:
        return _NumpyCompatUnpickler(handle).load()


class CustomVideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        video_names: Optional[Sequence[str]] = None,
        transform: Optional[callable] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        if video_names is None:
            self.video_folders = sorted(
                [name for name in os.listdir(root_dir) if name.endswith(".pkl")],
                key=extract_number,
            )
        else:
            normalized_names = [normalize_video_name(name) for name in video_names]
            self.video_folders = list(normalized_names)

        missing_paths = [
            os.path.join(self.root_dir, video_name)
            for video_name in self.video_folders
            if not os.path.isfile(os.path.join(self.root_dir, video_name))
        ]
        if missing_paths:
            preview = "\n  - ".join(missing_paths[:10])
            raise FileNotFoundError("Missing JIGSAWS PKL files:\n  - " + preview)

    def __len__(self) -> int:
        return len(self.video_folders)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray, str]:
        video_name = self.video_folders[idx]
        video_path = os.path.join(self.root_dir, video_name)
        video_data = _load_pickle_compat(video_path)

        features = video_data["feature"].astype("float32")
        error_labels = np.asarray(video_data["error_GT"]).astype(np.float32)
        video_length = len(error_labels)

        if self.transform:
            features = self.transform(features)

        return features, video_length, error_labels, video_name
