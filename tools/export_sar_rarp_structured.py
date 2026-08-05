#!/usr/bin/env python3
"""
Copy SAR-RARP videos and labels into a simpler per-video layout.
"""

import argparse
import json
import pickle
import re
import shutil
from pathlib import Path


VIDEO_FILENAME = "video_left.avi"
DISCRETE_LABEL_FILENAME = "action_discrete.txt"
CONTINUOUS_LABEL_FILENAME = "action_continuous.txt"
PKL_FILENAME = "dinov2_labels.pkl"
METADATA_FILENAME = "metadata.json"

SPLIT_CONFIG = {
    "train": {"gesture_dir": "gestures/train", "pkl_dir": "train_emb_DINOv2"},
    "test": {"gesture_dir": "gestures/test", "pkl_dir": "test_emb_DINOv2"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export SAR-RARP into a video-only structured layout.")
    parser.add_argument("--data-root", type=Path, default=Path("./data/SAR_RARP50"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_video_ids(root):
    return sorted(
        child.name for child in root.iterdir() if child.is_dir() and child.name.startswith("video_")
    )


def load_num_frames_from_continuous(path):
    last_end = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            _start, end, _label = line.split(",")
            last_end = int(end)
    if last_end is None:
        raise ValueError("No segments found in {0}".format(path))
    return last_end + 1


def pkl_max_frame_index(path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    image_names = data.get("image_name")
    if image_names is None or len(image_names) == 0:
        return None
    return max(int(Path(str(name)).stem) for name in image_names)


def candidate_pkl_paths(data_root, split, video_id):
    pkl_root = data_root / SPLIT_CONFIG[split]["pkl_dir"]
    candidates = []
    exact = pkl_root / "{0}.pkl".format(video_id)
    if exact.is_file():
        candidates.append(exact)
    match = re.fullmatch(r"(.+)_\d+", video_id)
    if match:
        alias = pkl_root / "{0}.pkl".format(match.group(1))
        if alias.is_file():
            candidates.append(alias)
    return candidates


def pick_pkl_path(data_root, split, video_id, num_frames):
    for path in candidate_pkl_paths(data_root, split, video_id):
        max_index = pkl_max_frame_index(path)
        if max_index is not None and max_index < num_frames:
            return path
    return None


def copy_video_folder(data_root, output_root, split, video_id, overwrite):
    src_dir = data_root / SPLIT_CONFIG[split]["gesture_dir"] / video_id
    dst_dir = output_root / split / video_id

    video_path = src_dir / VIDEO_FILENAME
    discrete_path = src_dir / DISCRETE_LABEL_FILENAME
    continuous_path = src_dir / CONTINUOUS_LABEL_FILENAME

    if not video_path.is_file():
        raise FileNotFoundError("Missing video file: {0}".format(video_path))
    if not discrete_path.is_file():
        raise FileNotFoundError("Missing discrete labels: {0}".format(discrete_path))
    if not continuous_path.is_file():
        raise FileNotFoundError("Missing continuous labels: {0}".format(continuous_path))

    num_frames = load_num_frames_from_continuous(continuous_path)
    pkl_path = pick_pkl_path(data_root, split, video_id, num_frames)

    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError("Destination already exists: {0}".format(dst_dir))
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, dst_dir / VIDEO_FILENAME)
    shutil.copy2(discrete_path, dst_dir / DISCRETE_LABEL_FILENAME)
    shutil.copy2(continuous_path, dst_dir / CONTINUOUS_LABEL_FILENAME)
    if pkl_path is not None:
        shutil.copy2(pkl_path, dst_dir / PKL_FILENAME)

    metadata = {
        "split": split,
        "video_id": video_id,
        "num_frames": num_frames,
        "video_path": str(video_path.resolve()),
        "has_error_labels": pkl_path is not None,
        "error_label_source": None if pkl_path is None else pkl_path.stem,
    }
    with (dst_dir / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def collect_video_ids(data_root, splits, requested_video_ids):
    requested = None if requested_video_ids is None else set(requested_video_ids)
    result = {}
    for split in splits:
        split_root = data_root / SPLIT_CONFIG[split]["gesture_dir"]
        video_ids = list_video_ids(split_root)
        if requested is not None:
            video_ids = [video_id for video_id in video_ids if video_id in requested]
        result[split] = video_ids
    return result


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    planned = collect_video_ids(data_root, args.splits, args.video_ids)

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for split in args.splits:
        print("[{0}] {1} videos".format(split, len(planned[split])))
        for video_id in planned[split]:
            src_dir = data_root / SPLIT_CONFIG[split]["gesture_dir"] / video_id
            num_frames = load_num_frames_from_continuous(src_dir / CONTINUOUS_LABEL_FILENAME)
            pkl_path = pick_pkl_path(data_root, split, video_id, num_frames)
            if args.dry_run:
                summary = {
                    "split": split,
                    "video_id": video_id,
                    "num_frames": num_frames,
                    "has_error_labels": pkl_path is not None,
                }
                manifest.append(summary)
                print(
                    "  - {0}: num_frames={1} has_error_labels={2}".format(
                        video_id, num_frames, pkl_path is not None
                    )
                )
            else:
                metadata = copy_video_folder(data_root, output_root, split, video_id, args.overwrite)
                manifest.append(metadata)
                print(
                    "  - exported {0}: num_frames={1} has_error_labels={2}".format(
                        video_id, metadata["num_frames"], metadata["has_error_labels"]
                    )
                )

    if not args.dry_run:
        with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"videos": manifest}, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
