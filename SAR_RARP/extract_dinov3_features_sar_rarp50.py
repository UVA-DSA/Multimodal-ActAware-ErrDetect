#!/usr/bin/env python3
"""
Extract DINOv3 frame features for SAR-RARP50.

Inputs:
  data/SAR_RARP50/{training_set,testing_set}/{video_id}/images/*.png

Outputs (same style as CLIP extractor):
  data/SAR_RARP50/vid_features/{model_tag}/{training_set,testing_set}/{video_id}/features.pt

Each features.pt is:
  Dict[int, Tensor(D)] mapping frame_index -> embedding vector
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


_FRAME_RE = re.compile(r"(\d+)\.png$")


def _list_frames(images_dir: str) -> List[Tuple[int, str]]:
    files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    out: List[Tuple[int, str]] = []
    for fp in files:
        m = _FRAME_RE.search(os.path.basename(fp))
        if not m:
            continue
        out.append((int(m.group(1)), fp))
    out.sort(key=lambda x: x[0])
    return out


def _sanitize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_")


@torch.no_grad()
def _extract_video_features(
    model,
    processor,
    images_dir: str,
    device: str,
    batch_size: int,
) -> Dict[int, torch.Tensor]:
    frames = _list_frames(images_dir)
    feats: Dict[int, torch.Tensor] = {}
    if len(frames) == 0:
        return feats

    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        imgs = []
        idxs = []
        for frame_idx, fp in chunk:
            try:
                imgs.append(Image.open(fp).convert("RGB"))
                idxs.append(frame_idx)
            except Exception as e:
                print(f"[WARN] Failed to read {fp}: {e}")
        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        # Prefer pooler_output; otherwise use CLS token embedding.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output.to(torch.float32)  # (B, D)
        else:
            emb = outputs.last_hidden_state[:, 0, :].to(torch.float32)  # (B, D)

        for j, frame_idx in enumerate(idxs):
            feats[frame_idx] = emb[j].detach().cpu()

    return feats


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 features for SAR_RARP50")
    parser.add_argument(
        "--hf_model",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face model id for DINOv3",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Output subdir tag under vid_features/ (default: hf_model with '/' replaced by '_')",
    )
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["training_set", "testing_set"],
        choices=["training_set", "testing_set"],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--max_videos", type=int, default=0, help="If >0, process at most this many videos per split")
    args = parser.parse_args()

    model_tag = args.model_tag or _sanitize_model_tag(args.hf_model)
    print(f"[INFO] data_root={args.data_root}")
    print(f"[INFO] hf_model={args.hf_model}")
    print(f"[INFO] model_tag={model_tag}")
    print(f"[INFO] splits={args.splits} batch_size={args.batch_size} device={args.device}")

    processor = AutoImageProcessor.from_pretrained(args.hf_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.hf_model, trust_remote_code=True).to(args.device).eval()

    total_videos = 0
    total_frames = 0
    for split in args.splits:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Missing split dir: {split_dir}")
            continue

        video_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        if args.max_videos and args.max_videos > 0:
            video_dirs = video_dirs[: args.max_videos]
        print(f"\n[INFO] Processing {split}: {len(video_dirs)} videos")

        for vid in tqdm(video_dirs, desc=f"Extracting {split}"):
            images_dir = os.path.join(split_dir, vid, "images")
            if not os.path.isdir(images_dir):
                continue

            out_path = os.path.join(args.data_root, "vid_features", model_tag, split, vid, "features.pt")
            if os.path.exists(out_path) and not args.force:
                continue

            feats = _extract_video_features(model, processor, images_dir, args.device, args.batch_size)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(feats, out_path)
            total_videos += 1
            total_frames += len(feats)

    print("\n[INFO] Done.")
    print(f"[INFO] Processed videos={total_videos} frames={total_frames}")
    print(f"[INFO] Output root: {os.path.join(args.data_root, 'vid_features', model_tag)}")


if __name__ == "__main__":
    main()
