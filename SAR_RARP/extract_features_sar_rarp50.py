#!/usr/bin/env python3
"""
Feature extraction for SAR-RARP50.

This mirrors `extract_features.py` (JIGSAWS) but targets SAR_RARP50 folder layout:
  data/SAR_RARP50/{training_set,testing_set}/{video}/images/*.png

Outputs:
  data/SAR_RARP50/vid_features/{model}/{training_set,testing_set}/{video}/features.pt

Each `features.pt` is a dict: frame_index(int) -> feature_tensor(D)

Notes:
- Existing ResNet50 features (`embed.npy`) already exist under each video folder.
- You can also pass `--finetuned_ckpt` to extract ResNet50 features from a SAR-RARP50
  error-label finetuned checkpoint and save the projected head features under a separate
  `vid_features/<name>/...` source.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from sar_rarp50_error_finetune_utils import load_sar_rarp50_error_checkpoint


AVAILABLE_MODELS = {
    "resnet18": {"type": "resnet", "feature_dim": 512, "description": "ResNet-18 (ImageNet)"},
    "resnet50": {"type": "resnet", "feature_dim": 2048, "description": "ResNet-50 (ImageNet)"},
    "resnet101": {"type": "resnet", "feature_dim": 2048, "description": "ResNet-101 (ImageNet)"},
    "clip-vit-base-patch32": {
        "type": "clip",
        "hf_name": "openai/clip-vit-base-patch32",
        "feature_dim": 768,
        "image_size": 224,
        "description": "CLIP ViT-B/32 (224x224)",
    },
    "clip-vit-base-patch16": {
        "type": "clip",
        "hf_name": "openai/clip-vit-base-patch16",
        "feature_dim": 768,
        "image_size": 224,
        "description": "CLIP ViT-B/16 (224x224)",
    },
    "clip-vit-large-patch14": {
        "type": "clip",
        "hf_name": "openai/clip-vit-large-patch14",
        "feature_dim": 1024,
        "image_size": 224,
        "description": "CLIP ViT-L/14 (224x224)",
    },
    "clip-vit-large-patch14-336": {
        "type": "clip",
        "hf_name": "openai/clip-vit-large-patch14-336",
        "feature_dim": 1024,
        "image_size": 336,
        "description": "CLIP ViT-L/14 (336x336)",
    },
}


def print_available_models():
    print("\nAvailable Models:")
    print("-" * 90)
    print(f"{'Model Name':<30} {'Feature Dim':<12} {'Description'}")
    print("-" * 90)
    for name, info in AVAILABLE_MODELS.items():
        print(f"{name:<30} {info['feature_dim']:<12} {info['description']}")
    print("-" * 90)
    print("Note: pass --finetuned_ckpt with --model resnet50 to extract from a SAR-RARP50 error-finetuned checkpoint.")


def get_resnet_extractor(model_name: str, device, finetuned_ckpt: str | None = None):
    if finetuned_ckpt is not None:
        if model_name != "resnet50":
            raise ValueError("--finetuned_ckpt is currently supported only with --model resnet50")
        model, ckpt_info = load_sar_rarp50_error_checkpoint(finetuned_ckpt, device)
        meta = ckpt_info.get("meta", {})
        backbone_name = meta.get("backbone", "resnet50")
        if backbone_name != model_name:
            raise ValueError(
                f"Checkpoint backbone mismatch: expected {model_name}, checkpoint reports {backbone_name}"
            )
        extractor = model.to(device)
        extractor.eval()
        return extractor

    if model_name == "resnet18":
        m = models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        m = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        m = models.resnet101(pretrained=True)
    else:
        raise ValueError(f"Unsupported ResNet type: {model_name}")
    m.fc = nn.Identity()
    m = m.to(device)
    m.eval()
    return m


def get_resnet_transform(image_size: int = 224, normalize: bool = True):
    ops = [
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def get_clip_extractor(model_name: str, device: str):
    from transformers import CLIPProcessor, CLIPVisionModel

    info = AVAILABLE_MODELS[model_name]
    vision = CLIPVisionModel.from_pretrained(info["hf_name"]).to(device)
    processor = CLIPProcessor.from_pretrained(info["hf_name"])
    vision.eval()
    return vision, processor


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


@torch.no_grad()
def extract_resnet(model, images_dir: str, out_path: str, device: str, batch_size: int, transform):
    frames = _list_frames(images_dir)
    if len(frames) == 0:
        return 0

    feats: Dict[int, torch.Tensor] = {}
    for i in range(0, len(frames), batch_size):
        chunk = frames[i : i + batch_size]
        imgs = []
        idxs = []
        for frame_idx, fp in chunk:
            try:
                img = Image.open(fp).convert("RGB")
                imgs.append(transform(img))
                idxs.append(frame_idx)
            except Exception as e:
                print(f"[WARN] Failed to read {fp}: {e}")
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)
        y = model(x)  # (B, D)
        if isinstance(y, (tuple, list)):
            y = y[0]
        for j, frame_idx in enumerate(idxs):
            feats[frame_idx] = y[j].cpu()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(feats, out_path)
    return len(feats)


@torch.no_grad()
def extract_clip(model, processor, images_dir: str, out_path: str, device: str, batch_size: int):
    frames = _list_frames(images_dir)
    if len(frames) == 0:
        return 0

    feats: Dict[int, torch.Tensor] = {}
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
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)
        y = outputs.pooler_output  # (B, D)
        for j, frame_idx in enumerate(idxs):
            feats[frame_idx] = y[j].cpu()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(feats, out_path)
    return len(feats)


def _resolve_output_model_name(args) -> str:
    if args.output_model_name:
        return args.output_model_name
    if args.finetuned_ckpt:
        return Path(args.finetuned_ckpt).stem
    return args.model


def main():
    parser = argparse.ArgumentParser(description="Extract SAR_RARP50 video frame features")
    parser.add_argument("--model", type=str, default="clip-vit-base-patch32", choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--finetuned_ckpt",
        type=str,
        default=None,
        help="Optional SAR-RARP50 finetuned ResNet50 checkpoint to use for extraction",
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        default=None,
        help="Override subdirectory name under vid_features when using --finetuned_ckpt (default: checkpoint stem)",
    )
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--splits", type=str, nargs="+", default=["training_set", "testing_set"], choices=["training_set", "testing_set"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--max_videos", type=int, default=0, help="If >0, process at most this many videos per split (debug)")
    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        return

    device = torch.device(args.device)
    info = AVAILABLE_MODELS[args.model]
    model_type = info["type"]
    output_model_name = _resolve_output_model_name(args)
    if args.finetuned_ckpt is not None and model_type != "resnet":
        raise ValueError("--finetuned_ckpt can only be used with ResNet models")
    print(f"[INFO] data_root={args.data_root}")
    if model_type == "resnet":
        extractor = get_resnet_extractor(args.model, device, finetuned_ckpt=args.finetuned_ckpt)
        transform = get_resnet_transform(image_size=224, normalize=True)
        processor = None
    else:
        extractor, processor = get_clip_extractor(args.model, device)
        transform = None

    feature_dim = getattr(extractor, "hidden_dim", info["feature_dim"])
    print(f"[INFO] model={args.model} type={model_type} dim={feature_dim}")
    print(f"[INFO] output_model_name={output_model_name}")
    if args.finetuned_ckpt:
        print(f"[INFO] finetuned_ckpt={args.finetuned_ckpt}")
    print(f"[INFO] splits={args.splits} batch_size={args.batch_size} device={device}")

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
            out_path = os.path.join(args.data_root, "vid_features", output_model_name, split, vid, "features.pt")
            if os.path.exists(out_path) and not args.force:
                continue
            if model_type == "resnet":
                n = extract_resnet(extractor, images_dir, out_path, device, args.batch_size, transform)
            else:
                n = extract_clip(extractor, processor, images_dir, out_path, device, args.batch_size)
            total_videos += 1
            total_frames += n

    print("\n[INFO] Done.")
    print(f"[INFO] Processed videos={total_videos} frames={total_frames}")
    print(f"[INFO] Output root: {os.path.join(args.data_root, 'vid_features', output_model_name)}")


if __name__ == "__main__":
    main()


