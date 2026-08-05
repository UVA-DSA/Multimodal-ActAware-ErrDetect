#!/usr/bin/env python3
"""
Generate ResNet50 ImageNet embeddings for SAR_RARP50 videos.

Saves to: data/SAR_RARP50/{split}/{video_id}/embed/embed.npy
Frames expected under: data/SAR_RARP50/{split}/{video_id}/images/frame_*.png
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


def _sorted_frame_paths(images_dir: str, number_regex: str | None) -> List[str]:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    names = [n for n in os.listdir(images_dir) if os.path.splitext(n)[1].lower() in image_exts]
    if len(names) == 0:
        return []
    if number_regex:
        num_re = re.compile(number_regex)
        frames = []
        for name in names:
            m = num_re.search(name)
            if m:
                frames.append((int(m.group(1)), os.path.join(images_dir, name)))
        if frames:
            frames.sort(key=lambda x: x[0])
            return [p for _, p in frames]
    # Fallback: lexicographic order
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _load_images(paths: List[str], transform) -> torch.Tensor:
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs, dim=0)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Create ResNet50 embed.npy for SAR_RARP50 videos.")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50", help="Root SAR_RARP50 data path")
    parser.add_argument("--split", type=str, default="testing_set", choices=["training_set", "testing_set"])
    parser.add_argument("--videos", type=str, default="video_46,video_49", help="Comma-separated video ids")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--number_regex", type=str, default=r"(\d+)", help="Regex to extract frame index")
    args = parser.parse_args()

    device = torch.device(args.device)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()

    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    if not video_ids:
        raise ValueError("No videos provided.")

    for vid in video_ids:
        images_dir = os.path.join(args.data_root, args.split, vid, "images")
        embed_dir = os.path.join(args.data_root, args.split, vid, "embed")
        embed_path = os.path.join(embed_dir, "embed.npy")
        _ensure_dir(embed_dir)

        frame_paths = _sorted_frame_paths(images_dir, args.number_regex)
        if len(frame_paths) == 0:
            raise FileNotFoundError(f"No frames found under {images_dir}")

        all_feats = []
        with torch.no_grad():
            for i in range(0, len(frame_paths), args.batch_size):
                batch_paths = frame_paths[i : i + args.batch_size]
                batch = _load_images(batch_paths, transform).to(device)
                feats = model(batch)  # (B, 2048)
                all_feats.append(feats.cpu().numpy().astype("float32"))

        embed = np.concatenate(all_feats, axis=0)
        np.save(embed_path, embed)
        print(f"[INFO] Saved {embed.shape} -> {embed_path}")


if __name__ == "__main__":
    main()
