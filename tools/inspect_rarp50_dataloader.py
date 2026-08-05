#!/usr/bin/env python3
"""
Inspect SAR-RARP50 dataloader windows and raw assets.

What this script shows for sampled windows:
- frame numbers
- error labels (raw frame-level from .pkl)
- gesture labels (best-effort from common .pkl keys)
- masks from collate function (padding mask)
- base and ges/tri feature stats and small value previews
- optional frame-strip image dumps

Base feature source notes:
- embed: `data_root/{training_set,testing_set}/{video}/embed/embed.npy`
- pt: `data_root/vid_features/{base_model}/{training_set,testing_set}/{video}/features.pt`
"""

import argparse
import os
import pickle
import random
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow running as:
#   python SAR_RARP/inspect_rarp50_dataloader.py ...
# from repo root, without installing SAR_RARP as a package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from SAR_RARP.dataloader_rarp50_features import (
        BaseFeatureSpec,
        SAR_RARP50DualContextDataset,
        collate_fn_dual_context,
    )
except Exception:
    from dataloader_rarp50_features import (
        BaseFeatureSpec,
        SAR_RARP50DualContextDataset,
        collate_fn_dual_context,
    )


def _parse_gesture_token(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if len(s) == 0:
            return -1
        if s.startswith("G") and s[1:].isdigit():
            return int(s[1:])
        if s.isdigit():
            return int(s)
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else -1
    return -1


def _normalize_gesture_vector(value: Any, target_len: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    if arr is None:
        return None

    if arr.ndim == 0:
        g = _parse_gesture_token(arr.item())
        return np.full((target_len,), g, dtype=np.int64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    out = np.array([_parse_gesture_token(v) for v in arr], dtype=np.int64)
    if out.size == target_len:
        return out
    if out.size > target_len:
        return out[:target_len]
    pad = np.full((target_len - out.size,), -1, dtype=np.int64)
    return np.concatenate([out, pad], axis=0)


def _extract_gesture_labels(video_data: Dict[str, Any], target_len: int) -> Tuple[np.ndarray, str]:
    candidate_keys = [
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
    for k in candidate_keys:
        if k in video_data:
            vec = _normalize_gesture_vector(video_data[k], target_len)
            if vec is not None:
                return vec, k
    return np.full((target_len,), -1, dtype=np.int64), "N/A"


def _choose_indices(n_total: int, n_samples: int, seed: int) -> List[int]:
    if n_total <= 0:
        return []
    n = min(n_total, n_samples)
    if n == 1:
        return [0]
    rng = random.Random(seed)
    if n_total <= n:
        idxs = list(range(n_total))
    else:
        # Evenly spread with tiny random jitter to avoid always deterministic same starts.
        base = np.linspace(0, n_total - 1, n).round().astype(int).tolist()
        idxs = []
        for b in base:
            lo = max(0, b - 2)
            hi = min(n_total - 1, b + 2)
            idxs.append(rng.randint(lo, hi))
        idxs = sorted(set(idxs))
        while len(idxs) < n:
            idxs.append(rng.randrange(0, n_total))
            idxs = sorted(set(idxs))
        idxs = idxs[:n]
    return idxs


def _feature_preview(x: torch.Tensor, n_dim: int = 6) -> str:
    if x.numel() == 0:
        return "[]"
    row = x[0] if x.dim() > 1 else x
    vals = row[:n_dim].detach().cpu().numpy().tolist()
    return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]"


def _save_frame_strip(
    frame_paths: Sequence[str],
    out_path: str,
    max_frames: int = 8,
) -> None:
    try:
        from PIL import Image, ImageOps, ImageDraw
    except Exception:
        return
    keep = [p for p in frame_paths if os.path.exists(p)]
    if len(keep) == 0:
        return
    if len(keep) > max_frames:
        ids = np.linspace(0, len(keep) - 1, max_frames).round().astype(int).tolist()
        keep = [keep[i] for i in ids]

    imgs = []
    for p in keep:
        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(img, (224, 224))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 80, 20), fill=(0, 0, 0))
        draw.text((4, 4), os.path.basename(p).replace(".png", ""), fill=(255, 255, 255))
        imgs.append(img)

    strip = Image.new("RGB", (224 * len(imgs), 224))
    for i, img in enumerate(imgs):
        strip.paste(img, (i * 224, 0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    strip.save(out_path)


def _load_pt_feature_frame_keys(data_root: str, model_name: str, split_set: str, video_id: str) -> List[int]:
    fp = os.path.join(data_root, "vid_features", model_name, split_set, video_id, "features.pt")
    d = torch.load(fp, map_location="cpu")
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict in {fp}, got {type(d)}")
    return sorted(int(k) for k in d.keys())


def _count_image_frames(data_root: str, split_set: str, video_id: str) -> int:
    img_dir = os.path.join(data_root, split_set, video_id, "images")
    if not os.path.isdir(img_dir):
        return 0
    return sum(1 for f in os.listdir(img_dir) if f.endswith(".png"))


def _count_base_feature_rows(data_root: str, base: BaseFeatureSpec, split_set: str, video_id: str) -> int:
    if base.source == "embed":
        fp = os.path.join(data_root, split_set, video_id, "embed", "embed.npy")
        if not os.path.exists(fp):
            return 0
        arr = np.load(fp, mmap_mode="r")
        return int(arr.shape[0]) if arr.ndim >= 1 else 0
    if base.source == "pt":
        if not base.model:
            return 0
        keys = _load_pt_feature_frame_keys(data_root, str(base.model), split_set, video_id)
        return len(keys)
    return 0


def inspect_split(
    data_root: str,
    pkl_dir: str,
    split_set: str,
    base: BaseFeatureSpec,
    gesture_embed: str,
    segment_length: int,
    step_size: int,
    num_windows: int,
    seed: int,
    out_dir: str,
    frame_index_base: int,
    label_mode: str,
) -> None:
    ds = SAR_RARP50DualContextDataset(
        data_root=data_root,
        pkl_dir=pkl_dir,
        split_set=split_set,
        base=base,
        gesture_embed=gesture_embed,
        video_ids=None,
        segment_length=segment_length,
        step_size=step_size,
        label_mode=label_mode,
    )
    if len(ds) == 0:
        print(f"[WARN] No windows in split={split_set}.")
        return

    chosen = _choose_indices(len(ds), num_windows, seed)
    raw_items = []
    info_rows = []
    pt_keys_cache: Dict[str, List[int]] = {}
    img_count_cache: Dict[str, int] = {}
    feat_count_cache: Dict[str, int] = {}

    for global_idx in chosen:
        vid, start = ds.indices[global_idx]
        item = ds[global_idx]
        if item is None:
            continue
        base_seg, ges_seg, _labels_vec, _vid = item
        raw_items.append((base_seg, ges_seg, torch.zeros(base_seg.size(0), dtype=torch.long), vid))

        pkl_path = os.path.join(pkl_dir, f"{vid}.pkl")
        with open(pkl_path, "rb") as f:
            video_data = pickle.load(f)
        err = np.asarray(video_data["error_GT"], dtype=np.int64)
        ges_all, ges_src = _extract_gesture_labels(video_data, target_len=int(err.size))

        t = int(base_seg.size(0))
        if vid not in img_count_cache:
            img_count_cache[vid] = _count_image_frames(data_root, split_set, vid)
        if vid not in feat_count_cache:
            feat_count_cache[vid] = _count_base_feature_rows(data_root, base, split_set, vid)

        # IMPORTANT:
        # - embed mode uses dense frame indexing (window index -> frame index)
        # - pt mode must map window index back to true frame-number keys in features.pt
        if base.source == "pt":
            if vid not in pt_keys_cache:
                pt_keys_cache[vid] = _load_pt_feature_frame_keys(data_root, str(base.model), split_set, vid)
            key_vec = pt_keys_cache[vid]
            feature_frame_nums = key_vec[start : start + t]
            err_win = np.array(
                [int(err[f]) if 0 <= int(f) < int(err.size) else -1 for f in feature_frame_nums],
                dtype=np.int64,
            )
            ges_win = np.array(
                [int(ges_all[f]) if 0 <= int(f) < int(ges_all.size) else -1 for f in feature_frame_nums],
                dtype=np.int64,
            )
        else:
            feature_frame_nums = [start + i for i in range(t)]
            err_win = err[start : start + t]
            ges_win = ges_all[start : start + t]

        image_frame_nums = [int(f) + int(frame_index_base) for f in feature_frame_nums]

        frame_paths = [
            os.path.join(data_root, split_set, vid, "images", f"{fn:09d}.png")
            for fn in image_frame_nums
        ]
        img_found = sum(int(os.path.exists(p)) for p in frame_paths)

        info_rows.append(
            {
                "global_idx": global_idx,
                "video_id": vid,
                "start": start,
                "len": t,
                "feature_frame_nums": feature_frame_nums,
                "image_frame_nums": image_frame_nums,
                "error_labels": err_win.tolist(),
                "gesture_labels": ges_win.tolist(),
                "gesture_source_key": ges_src,
                "img_found": img_found,
                "img_total": len(frame_paths),
                "img_folder_total": img_count_cache[vid],
                "video_image_folder_total": img_count_cache[vid],
                "video_base_feature_rows_total": feat_count_cache[vid],
                "base_shape": tuple(base_seg.shape),
                "ges_shape": tuple(ges_seg.shape),
                "base_mean": float(base_seg.mean().item()),
                "base_std": float(base_seg.std().item()),
                "ges_mean": float(ges_seg.mean().item()),
                "ges_std": float(ges_seg.std().item()),
                "base_preview": _feature_preview(base_seg),
                "ges_preview": _feature_preview(ges_seg),
                "frame_paths": frame_paths,
            }
        )

    # Build masks via collate to inspect actual dataloader mask behavior.
    collated = collate_fn_dual_context(raw_items)
    if collated is None:
        print(f"[WARN] Empty collate batch for split={split_set}.")
        return
    _base_padded, _ges_padded, masks, _labels_padded, vids = collated

    print("\n" + "=" * 88)
    print(f"[Split={split_set}] dataset_windows={len(ds)} sampled_windows={len(info_rows)} gesture_embed={gesture_embed}")
    print("=" * 88)
    for i, row in enumerate(info_rows):
        mask_row = masks[i].detach().cpu().numpy().astype(int).tolist()
        print(
            f"\n[{i}] global_idx={row['global_idx']} video={row['video_id']} start={row['start']} "
            f"len={row['len']} collate_vid={vids[i]}"
        )
        print(f"  feature_frame_nums: {row['feature_frame_nums']}")
        print(f"  image_frame_nums:   {row['image_frame_nums']}")
        print(f"  error_labels:    {row['error_labels']}")
        print(f"  gesture_labels:  {row['gesture_labels']} (source={row['gesture_source_key']})")
        print(f"  mask_row:        {mask_row}")
        print(f"  video_image_folder_total: {row['video_image_folder_total']}")
        print(f"  video_base_feature_rows_total: {row['video_base_feature_rows_total']}")
        print(
            f"  images_found:    {row['img_found']}/{row['img_total']} "
            f"(video_image_folder_total={row['img_folder_total']}, "
            f"example={row['frame_paths'][0] if row['frame_paths'] else 'N/A'})"
        )
        print(
            f"  base_shape={row['base_shape']} mean={row['base_mean']:.4f} std={row['base_std']:.4f} "
            f"preview={row['base_preview']}"
        )
        print(
            f"  {gesture_embed}_shape={row['ges_shape']} mean={row['ges_mean']:.4f} std={row['ges_std']:.4f} "
            f"preview={row['ges_preview']}"
        )

        strip_out = os.path.join(
            out_dir,
            split_set,
            f"{gesture_embed}_idx{row['global_idx']}_{row['video_id']}_s{row['start']}_framestrip.png",
        )
        _save_frame_strip(row["frame_paths"], strip_out, max_frames=8)
        print(f"  frame_strip_out: {strip_out}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SAR-RARP50 dataloader windows and assets.")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"])
    parser.add_argument("--base_feature_source", type=str, default="embed", choices=["embed", "pt"])
    parser.add_argument("--base_model", type=str, default=None, help="Required when --base_feature_source=pt")
    parser.add_argument("--gesture_embed", type=str, default="both", choices=["ges", "tri", "both"])
    parser.add_argument("--segment_length", type=int, default=40)
    parser.add_argument("--step_size", type=int, default=6)
    parser.add_argument("--label_mode", type=str, default="frame", choices=["segment", "frame"])
    parser.add_argument("--num_windows", type=int, default=6, help="Sampled windows per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./sar_rarp50_dataloader_debug")
    parser.add_argument("--frame_index_base", type=int, default=1, help="0 or 1 based image indexing")
    args = parser.parse_args()

    if args.base_feature_source == "pt" and not args.base_model:
        raise ValueError("--base_model is required when --base_feature_source=pt")
    if args.base_feature_source == "pt":
        base_root = os.path.join(args.data_root, "vid_features", args.base_model)
        print(f"[INFO] Using PT base features under: {base_root}")
        if not os.path.isdir(base_root):
            raise FileNotFoundError(
                f"Base feature directory not found: {base_root}. "
                "Expected PT features under data_root/vid_features/{base_model}/..."
            )
    base = (
        BaseFeatureSpec(source="embed", model=None)
        if args.base_feature_source == "embed"
        else BaseFeatureSpec(source="pt", model=args.base_model)
    )

    train_pkl_dir = os.path.join(args.data_root, "train_emb_DINOv2")
    test_pkl_dir = os.path.join(args.data_root, "test_emb_DINOv2")

    splits = []
    if args.split in {"train", "both"}:
        splits.append(("training_set", train_pkl_dir))
    if args.split in {"test", "both"}:
        splits.append(("testing_set", test_pkl_dir))

    gestures = ["ges", "tri"] if args.gesture_embed == "both" else [args.gesture_embed]
    for split_set, pkl_dir in splits:
        for ges_kind in gestures:
            inspect_split(
                data_root=args.data_root,
                pkl_dir=pkl_dir,
                split_set=split_set,
                base=base,
                gesture_embed=ges_kind,
                segment_length=args.segment_length,
                step_size=args.step_size,
                num_windows=args.num_windows,
                seed=args.seed,
                out_dir=args.out_dir,
                frame_index_base=args.frame_index_base,
                label_mode=args.label_mode,
            )


if __name__ == "__main__":
    main()

