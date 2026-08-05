#!/usr/bin/env python3
"""
Convert JIGSAWS Suturing / Needle_Passing videos into SAR-RARP-style PKLs.

Each output PKL contains one labeled frame sequence per video with keys:
  - feature:   (N, 2048) float32 ResNet50 features
  - error_GT:  (N,) int64 frame-wise error labels
  - gesture_GT:(N,) int64 frame-wise gesture ids
  - image_name:(N,) frame file basenames

The JIGSAWS error annotations are segment-level. This script expands every
segment label to all frames whose frame number falls inside that segment's
inclusive [start_time, end_time] interval. Frames outside annotated intervals
are skipped because they do not have a segment error label.

Important timing note:
- The PNGs under `data/vid_frames/...` are already downsampled from 30 Hz to
  10 Hz, but they keep the original frame numbering (`0, 3, 6, ...`).
- The annotation CSVs are still expressed in that original frame-number space.
- This script therefore compares the saved frame numbers directly against the
  annotation intervals and does not renumber frames to `0, 1, 2, ...`.

Examples
--------
Reuse existing ResNet50 frame features when available, otherwise extract them:
  python convert_jigsaws_to_sar_rarp_pkl.py

Force fresh ResNet50 extraction from raw frames:
  python convert_jigsaws_to_sar_rarp_pkl.py --feature_mode extract

Convert a single video:
  python convert_jigsaws_to_sar_rarp_pkl.py --videos Suturing_S02_T01
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


TASK_ALIASES = {
    "suturing": "Suturing",
    "needle_passing": "Needle_Passing",
    "needle-passing": "Needle_Passing",
    "needle passing": "Needle_Passing",
}

FRAME_PATTERN = re.compile(r"frame_(\d+)\.png$")


@dataclass(frozen=True)
class SegmentInterval:
    start_time: int
    end_time: int
    gesture_id: int
    error_label: int


def normalize_task_name(task_name: str) -> str:
    key = task_name.strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError(
            f"Unsupported task '{task_name}'. Expected one of: {sorted(TASK_ALIASES.values())}"
        )
    return TASK_ALIASES[key]


def parse_gesture_id(raw_value: str) -> int:
    value = str(raw_value).strip()
    if value.startswith("G"):
        return int(value[1:])
    return int(value)


def list_videos_for_task(data_root: str, task_name: str) -> List[str]:
    error_dir = os.path.join(data_root, task_name, "errors")
    if not os.path.isdir(error_dir):
        raise FileNotFoundError(f"Missing error directory: {error_dir}")
    return sorted(os.path.splitext(name)[0] for name in os.listdir(error_dir) if name.endswith(".csv"))


def load_segments(error_csv_path: str) -> List[SegmentInterval]:
    segments: List[SegmentInterval] = []
    with open(error_csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"start_time", "end_time", "gesture", "error1_nor0"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns {sorted(missing_columns)} in {error_csv_path}")

        for row in reader:
            start_time = int(row["start_time"])
            end_time = int(row["end_time"])
            if end_time < start_time:
                raise ValueError(
                    f"Invalid interval [{start_time}, {end_time}] in {error_csv_path}"
                )
            segments.append(
                SegmentInterval(
                    start_time=start_time,
                    end_time=end_time,
                    gesture_id=parse_gesture_id(row["gesture"]),
                    error_label=int(row["error1_nor0"]),
                )
            )

    segments.sort(key=lambda segment: (segment.start_time, segment.end_time))
    return segments


def list_frame_info(frames_dir: str) -> List[Tuple[int, str, str]]:
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    frame_info: List[Tuple[int, str, str]] = []
    for frame_path in frame_paths:
        frame_name = os.path.basename(frame_path)
        match = FRAME_PATTERN.search(frame_name)
        if not match:
            continue
        frame_number = int(match.group(1))
        frame_info.append((frame_number, frame_name, frame_path))
    if not frame_info:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")
    return frame_info


def build_resnet50(device: torch.device) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def get_resnet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_precomputed_feature_dict(feature_path: str) -> Dict[int, np.ndarray]:
    if not os.path.isfile(feature_path):
        raise FileNotFoundError(f"Missing precomputed features: {feature_path}")
    feature_dict = torch.load(feature_path, map_location="cpu")
    output: Dict[int, np.ndarray] = {}
    for frame_number, feature in feature_dict.items():
        output[int(frame_number)] = np.asarray(feature, dtype=np.float32)
    return output


def extract_feature_dict(
    frame_info: Sequence[Tuple[int, str, str]],
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    feature_dict: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start_idx in range(0, len(frame_info), batch_size):
            batch = frame_info[start_idx : start_idx + batch_size]
            images = []
            frame_numbers = []
            for frame_number, _, frame_path in batch:
                image = Image.open(frame_path).convert("RGB")
                images.append(transform(image))
                frame_numbers.append(frame_number)
            if not images:
                continue
            image_tensor = torch.stack(images, dim=0).to(device)
            features = model(image_tensor).detach().cpu().numpy().astype(np.float32)
            for frame_number, feature in zip(frame_numbers, features):
                feature_dict[int(frame_number)] = feature
    return feature_dict


def resolve_feature_dict(
    *,
    feature_mode: str,
    feature_path: str,
    frame_info: Sequence[Tuple[int, str, str]],
    model: nn.Module | None,
    transform: transforms.Compose | None,
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[int, np.ndarray], str]:
    if feature_mode in {"auto", "reuse"} and os.path.isfile(feature_path):
        return load_precomputed_feature_dict(feature_path), "reuse"
    if feature_mode == "reuse":
        raise FileNotFoundError(f"Requested feature reuse but file is missing: {feature_path}")
    if model is None or transform is None:
        raise ValueError("Model and transform are required for feature extraction.")
    return extract_feature_dict(frame_info, model, transform, device, batch_size), "extract"


def build_video_arrays(
    frame_info: Sequence[Tuple[int, str, str]],
    segments: Sequence[SegmentInterval],
    feature_dict: Dict[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    features: List[np.ndarray] = []
    error_labels: List[int] = []
    gesture_labels: List[int] = []
    image_names: List[str] = []

    skipped_unlabeled = 0
    skipped_missing_features = 0
    segment_index = 0

    for frame_number, frame_name, _ in frame_info:
        while segment_index < len(segments) and frame_number > segments[segment_index].end_time:
            segment_index += 1
        if segment_index >= len(segments):
            skipped_unlabeled += 1
            continue

        segment = segments[segment_index]
        if frame_number < segment.start_time:
            skipped_unlabeled += 1
            continue

        feature = feature_dict.get(frame_number)
        if feature is None:
            skipped_missing_features += 1
            continue

        features.append(np.asarray(feature, dtype=np.float32))
        error_labels.append(int(segment.error_label))
        gesture_labels.append(int(segment.gesture_id))
        image_names.append(frame_name)

    if not features:
        raise ValueError("No labeled frames were collected for this video.")

    video_arrays = {
        "feature": np.stack(features, axis=0).astype(np.float32),
        "error_GT": np.asarray(error_labels, dtype=np.int64),
        "gesture_GT": np.asarray(gesture_labels, dtype=np.int64),
        "image_name": np.asarray(image_names),
    }
    video_arrays["_skipped_unlabeled"] = np.asarray([skipped_unlabeled], dtype=np.int64)
    video_arrays["_skipped_missing_features"] = np.asarray([skipped_missing_features], dtype=np.int64)
    return video_arrays


def save_video_pickle(output_path: str, video_arrays: Dict[str, np.ndarray]) -> None:
    payload = {
        "feature": video_arrays["feature"],
        "error_GT": video_arrays["error_GT"],
        "gesture_GT": video_arrays["gesture_GT"],
        "image_name": video_arrays["image_name"],
    }
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def iter_requested_videos(
    data_root: str,
    tasks: Sequence[str],
    requested_videos: Sequence[str] | None,
) -> Iterable[Tuple[str, str]]:
    task_to_videos = {task: set(list_videos_for_task(data_root, task)) for task in tasks}

    if requested_videos:
        for video_name in requested_videos:
            matched_task = None
            for task in tasks:
                if video_name in task_to_videos[task]:
                    matched_task = task
                    break
            if matched_task is None:
                raise FileNotFoundError(
                    f"Could not find video '{video_name}' under tasks {list(tasks)}"
                )
            yield matched_task, video_name
        return

    for task in tasks:
        for video_name in sorted(task_to_videos[task]):
            yield task, video_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JIGSAWS videos into SAR-RARP-style PKLs")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Suturing", "Needle_Passing"],
        help="Tasks to convert. Supported: Suturing, Needle_Passing",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Optional explicit list of video ids to convert (for example Suturing_S02_T01).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/jigsaws_sar_rarp_pkls/resnet50",
        help="Directory where flat per-video PKLs will be written.",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["auto", "extract", "reuse"],
        default="auto",
        help="Reuse precomputed resnet50 features when possible, or extract from frames.",
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        default=None,
        help="Root containing precomputed resnet50 features. Defaults to {data_root}/vid_features/resnet50.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    tasks = [normalize_task_name(task_name) for task_name in args.tasks]
    output_root = os.path.abspath(args.output_root)
    feature_root = os.path.abspath(
        args.feature_root or os.path.join(args.data_root, "vid_features", "resnet50")
    )
    os.makedirs(output_root, exist_ok=True)

    device = torch.device(args.device)
    model = None
    transform = None
    if args.feature_mode in {"auto", "extract"}:
        model = build_resnet50(device)
        transform = get_resnet_transform()

    converted = 0
    skipped_existing = 0
    for task_name, video_name in iter_requested_videos(args.data_root, tasks, args.videos):
        output_path = os.path.join(output_root, f"{video_name}.pkl")
        if os.path.exists(output_path) and not args.force:
            skipped_existing += 1
            print(f"[SKIP] {video_name}: output already exists at {output_path}")
            continue

        error_csv_path = os.path.join(args.data_root, task_name, "errors", f"{video_name}.csv")
        frames_dir = os.path.join(args.data_root, "vid_frames", task_name, video_name)
        precomputed_feature_path = os.path.join(feature_root, task_name, video_name, "features.pt")

        segments = load_segments(error_csv_path)
        frame_info = list_frame_info(frames_dir)
        feature_dict, feature_source = resolve_feature_dict(
            feature_mode=args.feature_mode,
            feature_path=precomputed_feature_path,
            frame_info=frame_info,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
        )
        video_arrays = build_video_arrays(frame_info, segments, feature_dict)
        save_video_pickle(output_path, video_arrays)

        converted += 1
        skipped_unlabeled = int(video_arrays["_skipped_unlabeled"][0])
        skipped_missing = int(video_arrays["_skipped_missing_features"][0])
        print(
            "[OK] {} | task={} | source={} | kept_frames={} | skipped_unlabeled={} | skipped_missing_features={} | out={}".format(
                video_name,
                task_name,
                feature_source,
                len(video_arrays["error_GT"]),
                skipped_unlabeled,
                skipped_missing,
                output_path,
            )
        )

    print(
        "\nDone. converted_videos={} skipped_existing={} output_root={}".format(
            converted,
            skipped_existing,
            output_root,
        )
    )


if __name__ == "__main__":
    main()
