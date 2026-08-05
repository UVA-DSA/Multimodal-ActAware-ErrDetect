#!/usr/bin/env python3
"""
Inspect gesture-label distributions from SAR-RARP50 .pkl files.

This script samples PKL files from:
  data_root/train_emb_DINOv2
  data_root/test_emb_DINOv2

and reports:
- which gesture-like key was found per file
- set of unique gesture values
- counts of each gesture value
- split-level and overall aggregate distributions
"""

import argparse
import os
import pickle
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


GESTURE_CANDIDATE_KEYS = [
    "gesture_GT",
    "gesture_gt",
    "gesture",
    "gestures",
    "gesture_label",
    "gesture_labels",
    "ges_GT",
    "ges_gt",
    "transcript",
    "transcriptions",
]


def _normalize_gesture_token(x: Any) -> Optional[str]:
    """Convert gesture value to a canonical string label (e.g., G1, G2)."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # float 3.0 -> G3, float 3.2 -> G3.2
        if float(x).is_integer():
            return f"G{int(x)}"
        return f"G{x}"
    if isinstance(x, str):
        s = x.strip()
        if len(s) == 0:
            return None
        if re.fullmatch(r"[Gg]\d+", s):
            return f"G{int(s[1:])}"
        if re.fullmatch(r"\d+", s):
            return f"G{int(s)}"
        # Keep as-is for non-standard labels (e.g., "needle_pass")
        return s
    return None


def _flatten_gesture_values(obj: Any) -> List[str]:
    """
    Recursively flatten gesture-like values from nested lists/tuples/dicts.
    Handles common transcript formats and generic nested containers.
    """
    out: List[str] = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        # Common transcript dict patterns may include gesture fields.
        for k, v in obj.items():
            lk = str(k).lower()
            if "gesture" in lk or lk in {"g", "label", "name", "id"}:
                out.extend(_flatten_gesture_values(v))
            else:
                # still recurse to be robust on unknown nested schema
                out.extend(_flatten_gesture_values(v))
        return out

    if isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_flatten_gesture_values(item))
        return out

    tok = _normalize_gesture_token(obj)
    if tok is not None:
        out.append(tok)
    return out


def _extract_gesture_labels_from_pkl(video_data: Dict[str, Any]) -> Tuple[List[str], str]:
    for key in GESTURE_CANDIDATE_KEYS:
        if key in video_data:
            labels = _flatten_gesture_values(video_data[key])
            return labels, key
    return [], "N/A"


def _sample_files(pkl_dir: str, max_files: int, seed: int) -> List[str]:
    files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
    files.sort()
    if max_files <= 0 or max_files >= len(files):
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, max_files))


def _print_counter(title: str, counter: Counter) -> None:
    print(f"\n{title}")
    if len(counter) == 0:
        print("  (no gesture labels found)")
        return
    total = sum(counter.values())
    print(f"  total labels: {total}")
    print(f"  unique labels ({len(counter)}): {sorted(counter.keys())}")
    for label, cnt in counter.most_common():
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  {label:>8s}: {cnt:6d} ({pct:6.2f}%)")


def inspect_split(split_name: str, pkl_dir: str, max_files: int, seed: int) -> Counter:
    chosen = _sample_files(pkl_dir, max_files=max_files, seed=seed)
    print("\n" + "=" * 96)
    print(f"[{split_name}] pkl_dir={pkl_dir}")
    print(f"[{split_name}] sampled_files={len(chosen)}")
    print("=" * 96)

    split_counter: Counter = Counter()

    for fn in chosen:
        fp = os.path.join(pkl_dir, fn)
        with open(fp, "rb") as f:
            data = pickle.load(f)
        labels, src_key = _extract_gesture_labels_from_pkl(data)
        c = Counter(labels)
        split_counter.update(c)

        uniques = sorted(c.keys())
        preview = ", ".join(uniques[:12]) if len(uniques) > 0 else "none"
        print(
            f"{fn:24s} key={src_key:14s} labels={len(labels):6d} "
            f"unique={len(uniques):3d} [{preview}{' ...' if len(uniques) > 12 else ''}]"
        )

    _print_counter(f"[{split_name}] aggregate gesture distribution", split_counter)
    return split_counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect gesture distributions in SAR-RARP50 PKL files.")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"])
    parser.add_argument("--max_files", type=int, default=10, help="Max PKL files per split (<=0 means all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_root, "train_emb_DINOv2")
    test_dir = os.path.join(args.data_root, "test_emb_DINOv2")

    if args.split in {"train", "both"} and not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing directory: {train_dir}")
    if args.split in {"test", "both"} and not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing directory: {test_dir}")

    global_counter: Counter = Counter()
    if args.split in {"train", "both"}:
        global_counter.update(inspect_split("train", train_dir, max_files=args.max_files, seed=args.seed))
    if args.split in {"test", "both"}:
        global_counter.update(inspect_split("test", test_dir, max_files=args.max_files, seed=args.seed + 1))

    print("\n" + "#" * 96)
    _print_counter("[overall] aggregate gesture distribution", global_counter)
    print("#" * 96)


if __name__ == "__main__":
    main()

