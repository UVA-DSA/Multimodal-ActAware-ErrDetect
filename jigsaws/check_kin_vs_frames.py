#!/usr/bin/env python3
"""
Compare kinematics CSV row counts vs image frame counts for selected JIGSAWS videos.

Example:
  python check_kin_vs_frames.py --task Needle_Passing --videos Needle_Passing_S02_T01,Needle_Passing_S03_T01
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import pandas as pd


def _resolve_kin_path(data_root: str, task: str, video_id: str) -> str:
    candidates = [
        os.path.join(data_root, task, "kinematics", video_id),
        os.path.join(data_root, task, "kinematics", f"{video_id}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No kinematics file found for {video_id}. Tried: {candidates}")


def _count_kin_rows(kin_path: str) -> int:
    return len(pd.read_csv(kin_path))


def _count_frames(data_root: str, task: str, video_id: str) -> int:
    frame_glob = os.path.join(data_root, "vid_frames", task, video_id, "*.png")
    return len(glob.glob(frame_glob))


def _parse_videos(videos_arg: str) -> List[str]:
    return [v.strip() for v in videos_arg.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compare kinematics rows vs image frames")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--task", type=str, required=True, choices=["Suturing", "Needle_Passing"])
    parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="Comma-separated video ids, e.g. Needle_Passing_S02_T01,Needle_Passing_S03_T01",
    )
    args = parser.parse_args()

    videos = _parse_videos(args.videos)
    if not videos:
        raise ValueError("No videos provided.")

    print(f"{'video_id':<28} {'kin_rows':>10} {'img_frames':>10} {'diff(kin-img)':>14} {'kin/img':>10}")
    print("-" * 78)

    for vid in videos:
        kin_path = _resolve_kin_path(args.data_root, args.task, vid)
        kin_rows = _count_kin_rows(kin_path)
        img_frames = _count_frames(args.data_root, args.task, vid)
        diff = kin_rows - img_frames
        ratio = (kin_rows / img_frames) if img_frames > 0 else float("inf")
        ratio_str = f"{ratio:.3f}" if img_frames > 0 else "inf"
        print(f"{vid:<28} {kin_rows:>10d} {img_frames:>10d} {diff:>14d} {ratio_str:>10}")


if __name__ == "__main__":
    main()
