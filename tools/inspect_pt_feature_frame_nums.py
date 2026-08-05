#!/usr/bin/env python3
"""
Inspect available frame numbers in SAR-RARP50 PT feature files.

Expected PT path layout:
  data_root/vid_features/{model}/{split_set}/{video_id}/features.pt

`features.pt` is expected to be a dict[int -> tensor].
The dict keys are the frame numbers available for that video/model.
"""

import argparse
import os
from typing import Dict, List

import torch


def _discover_video_ids(model_root: str, split_set: str) -> List[str]:
    split_root = os.path.join(model_root, split_set)
    if not os.path.isdir(split_root):
        return []
    vids = [d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))]
    return sorted(vids)


def _load_feature_dict(path: str) -> Dict[int, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    # normalize keys to int if possible
    out: Dict[int, torch.Tensor] = {}
    for k, v in obj.items():
        out[int(k)] = v
    return out


def _summarize_frame_nums(frame_nums: List[int]) -> None:
    if len(frame_nums) == 0:
        print("No frame numbers found.")
        return

    frame_nums = sorted(frame_nums)
    print(f"Total frames in feature file: {len(frame_nums)}")
    print(f"Min frame: {frame_nums[0]}")
    print(f"Max frame: {frame_nums[-1]}")
    print(f"First 20 frame nums: {frame_nums[:20]}")
    print(f"Last 20 frame nums: {frame_nums[-20:]}")

    gaps = []
    for a, b in zip(frame_nums[:-1], frame_nums[1:]):
        if b != a + 1:
            gaps.append((a, b))
    if len(gaps) == 0:
        print("Frame numbers are contiguous (no gaps).")
    else:
        print(f"Found {len(gaps)} non-contiguous jumps. Showing first 20:")
        for a, b in gaps[:20]:
            print(f"  jump: {a} -> {b} (missing {b - a - 1} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect frame numbers in SAR-RARP50 PT features.")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--model", type=str, default="clip-vit-base-patch16")
    parser.add_argument("--split_set", type=str, default="training_set", choices=["training_set", "testing_set"])
    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help="Video id folder name (e.g., video_01). If omitted, the first available video in split is used.",
    )
    args = parser.parse_args()

    model_root = os.path.join(args.data_root, "vid_features", args.model)
    if not os.path.isdir(model_root):
        raise FileNotFoundError(f"Model root not found: {model_root}")

    if args.video_id is None:
        vids = _discover_video_ids(model_root, args.split_set)
        if len(vids) == 0:
            raise FileNotFoundError(f"No videos found under: {os.path.join(model_root, args.split_set)}")
        video_id = vids[0]
        print(f"[INFO] --video_id not provided. Using first available: {video_id}")
    else:
        video_id = args.video_id

    feat_path = os.path.join(model_root, args.split_set, video_id, "features.pt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    feats = _load_feature_dict(feat_path)
    frame_nums = sorted(list(feats.keys()))

    print("=" * 88)
    print(f"Model:      {args.model}")
    print(f"Split:      {args.split_set}")
    print(f"Video ID:   {video_id}")
    print(f"File:       {feat_path}")
    if len(frame_nums) > 0:
        feat_dim = int(feats[frame_nums[0]].shape[0]) if hasattr(feats[frame_nums[0]], "shape") else -1
        print(f"Feature dim: {feat_dim}")
    print("=" * 88)
    _summarize_frame_nums(frame_nums)


if __name__ == "__main__":
    main()

