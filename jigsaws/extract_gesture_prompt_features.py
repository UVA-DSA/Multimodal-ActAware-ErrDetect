#!/usr/bin/env python3
"""
Extract per-frame hidden features from a finetuned JIGSAWS prompt model checkpoint.

Outputs per-video dict:
  frame_number (int) -> feature tensor (hidden_dim,)

Default output structure:
  gesture ckpt -> ./data/gesture_prompt_features/{ckpt_stem}/{task}/{video_name}/features.pt
  error ckpt   -> ./data/error_prompt_features/{ckpt_stem}/{task}/{video_name}/features.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm

from finetune_gesture_prompt_model import AVAILABLE_BACKBONES, build_transforms
from jigsaws_splits import get_split_fold_ids, resolve_split_root
from prompt_finetune_models import GestureFinetuneConfig, apply_lora_to_clip_vision, build_gesture_finetune_model, resolve_num_classes


def _load_ckpt(ckpt_path: str, device: torch.device) -> Tuple[Dict, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"]
    meta = ckpt.get("meta", {})
    args = ckpt.get("args", {})
    return state, {"meta": meta, "args": args, "ckpt": ckpt}


def _state_dict_has_lora(state: Dict) -> bool:
    # Our LoRALinear stores weights under *.lora_A.weight / *.lora_B.weight
    return any(".lora_A." in k or ".lora_B." in k for k in state.keys())


def _infer_num_classes_from_state(state: Dict) -> Optional[int]:
    weight = state.get("classifier.weight")
    if torch.is_tensor(weight) and weight.ndim >= 2:
        return int(weight.size(0))
    bias = state.get("classifier.bias")
    if torch.is_tensor(bias) and bias.ndim >= 1:
        return int(bias.size(0))
    return None


def _infer_resnet_projected_dim_from_state(state: Dict) -> Optional[int]:
    proj_weight = state.get("feature_proj.0.weight")
    if torch.is_tensor(proj_weight) and proj_weight.ndim >= 2:
        return int(proj_weight.size(0))
    return None


def _resolve_label_info(meta: Dict, ckpt_args: Dict, state: Dict) -> Tuple[str, int, Optional[int]]:
    num_classes = meta.get("num_classes") or ckpt_args.get("num_classes") or _infer_num_classes_from_state(state)
    label_type = meta.get("label_type") or ckpt_args.get("label_type")
    projected_dim = meta.get("projected_dim") or ckpt_args.get("resnet_projected_dim") or _infer_resnet_projected_dim_from_state(state)

    if label_type is None:
        if num_classes == 2:
            label_type = "error"
        elif num_classes == 8 or num_classes is None:
            # Backward compatibility for older gesture-only checkpoints.
            label_type = "gesture"
        else:
            raise ValueError(f"Could not infer label_type from checkpoint metadata (num_classes={num_classes})")

    resolved_num_classes = resolve_num_classes(label_type, num_classes=num_classes)
    return str(label_type), int(resolved_num_classes), (int(projected_dim) if projected_dim is not None else None)


def _resolve_fold_ckpts(
    ckpt_arg: str,
    folds: Tuple[int, ...] = (1, 2, 3, 4, 5),
    backbone_filter: Optional[str] = None,
) -> Tuple[Dict[int, str], str]:
    if os.path.isdir(ckpt_arg):
        ckpt_map: Dict[int, str] = {}
        for fold in folds:
            if backbone_filter:
                pattern = os.path.join(ckpt_arg, f"*{backbone_filter}*_fold{fold}_best.pth")
            else:
                pattern = os.path.join(ckpt_arg, f"*_fold{fold}_best.pth")
            matches = sorted(glob.glob(pattern))
            if len(matches) != 1:
                raise ValueError(f"Expected 1 ckpt for fold {fold} under {ckpt_arg}, found {len(matches)}")
            ckpt_map[fold] = matches[0]
        group_name = Path(ckpt_arg).name
        if backbone_filter:
            group_name = f"{group_name}_{backbone_filter}"
        return ckpt_map, group_name

    if os.path.isfile(ckpt_arg):
        ckpt_path = Path(ckpt_arg)
        stem = ckpt_path.stem
        m = re.match(r"(.+)_fold(\d+)_(.+)", stem)
        if m:
            base, _, tail = m.groups()
            ckpt_map = {}
            for fold in folds:
                candidate = ckpt_path.with_name(f"{base}_fold{fold}_{tail}{ckpt_path.suffix}")
                if candidate.exists():
                    ckpt_map[fold] = str(candidate)
            if len(ckpt_map) == len(folds):
                return ckpt_map, f"{base}_{tail}"
        return {1: str(ckpt_path)}, stem

    raise FileNotFoundError(f"Checkpoint path not found: {ckpt_arg}")


def _load_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    backbone_override: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    state, ckpt_info = _load_ckpt(ckpt_path, device)
    meta = ckpt_info["meta"]
    ckpt_args = ckpt_info["args"]
    backbone = backbone_override or meta.get("backbone")
    if backbone is None:
        raise ValueError("Could not infer backbone from ckpt meta; pass --backbone")

    label_type, num_classes, projected_dim = _resolve_label_info(meta, ckpt_args, state)
    cfg = GestureFinetuneConfig(
        backbone=backbone,
        label_type=label_type,
        num_classes=num_classes,
        resnet_projected_dim=projected_dim,
    )
    model, _ = build_gesture_finetune_model(cfg)

    # If this is a CLIP+LoRA checkpoint, rebuild the model with LoRA-wrapped attention projections
    # so the state_dict keys match (....q_proj.base..., ....q_proj.lora_A..., etc.).
    if backbone.startswith("clip-vit") and _state_dict_has_lora(state):
        lora_r = int(ckpt_args.get("lora_r", 8))
        lora_alpha = int(ckpt_args.get("lora_alpha", 16))
        lora_dropout = float(ckpt_args.get("lora_dropout", 0.0))
        apply_lora_to_clip_vision(model.vision, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    return model, {
        "backbone": backbone,
        "label_type": label_type,
        "num_classes": num_classes,
        "projected_dim": projected_dim,
    }


def _load_fold_video_map(split_root: str, task: str, folds: Tuple[int, ...] = (1, 2, 3, 4, 5)) -> Dict[str, int]:
    video_to_fold: Dict[str, int] = {}
    for fold in folds:
        test_csv = os.path.join(split_root, task, f"{fold}out", "test.csv")
        if not os.path.exists(test_csv):
            print(f"[WARN] Missing split test list: {test_csv}")
            continue
        with open(test_csv, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                name = os.path.splitext(os.path.basename(raw))[0]
                if name in video_to_fold and video_to_fold[name] != fold:
                    print(f"[WARN] Video listed in multiple folds: {name}")
                    continue
                video_to_fold[name] = fold
    return video_to_fold


@torch.no_grad()
def extract_video_features(model, transform, video_dir: str, output_path: str, device: torch.device, batch_size: int = 64):
    frame_pattern = re.compile(r"frame_(\d+)\.png")
    frame_files = sorted(glob.glob(os.path.join(video_dir, "frame_*.png")))
    if len(frame_files) == 0:
        return 0

    frame_info = []
    for fp in frame_files:
        m = frame_pattern.search(os.path.basename(fp))
        if not m:
            continue
        frame_info.append((int(m.group(1)), fp))
    frame_info.sort(key=lambda x: x[0])

    feats: Dict[int, torch.Tensor] = {}
    model.eval()

    for i in range(0, len(frame_info), batch_size):
        batch = frame_info[i : i + batch_size]
        imgs = []
        nums = []
        for n, fp in batch:
            try:
                img = Image.open(fp).convert("RGB")
                img_t = transform(img)
            except Exception:
                continue
            imgs.append(img_t)
            nums.append(n)
        if len(imgs) == 0:
            continue

        x = torch.stack(imgs).to(device)  # (B,C,H,W)
        # model expects (B,T,C,H,W)
        x = x.unsqueeze(1)
        hidden, _ = model(x)  # hidden: (B,1,D)
        hidden = hidden.squeeze(1).cpu()
        for j, n in enumerate(nums):
            feats[n] = hidden[j]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(feats, output_path)
    return len(feats)


def main():
    parser = argparse.ArgumentParser(description="Extract hidden features from finetuned gesture/error prompt model")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint .pth, or directory containing *_fold{1..5}_best.pth",
    )
    parser.add_argument("--backbone", type=str, default=None, choices=AVAILABLE_BACKBONES, help="Backbone override (else read from ckpt meta)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--split_scheme", type=str, default="loso", choices=["loso", "louo"])
    parser.add_argument(
        "--split_root",
        "--loso_root",
        dest="split_root",
        type=str,
        default=None,
        help="Root containing split CSVs. Defaults to ./LOSO or ./LOUO from --split_scheme.",
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=["Suturing", "Needle_Passing"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root dir (default: data/{gesture_prompt_features|error_prompt_features}/{ckpt_stem})",
    )
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    fold_ids = get_split_fold_ids(args.split_scheme)
    args.split_root = resolve_split_root(args.split_scheme, split_root=args.split_root, repo_root=".")
    fold_ckpts, ckpt_group = _resolve_fold_ckpts(args.ckpt, folds=fold_ids, backbone_filter=args.backbone)
    use_fold_ckpts = len(fold_ckpts) > 1

    ckpt_stem = ckpt_group
    model_cache: Dict[int, torch.nn.Module] = {}
    backbone: Optional[str] = None
    label_type: Optional[str] = None
    num_classes: Optional[int] = None
    transform = None

    if not use_fold_ckpts:
        model, model_info = _load_model_from_ckpt(fold_ckpts[1], device, args.backbone)
        model_cache[1] = model
        backbone = str(model_info["backbone"])
        label_type = str(model_info["label_type"])
        num_classes = int(model_info["num_classes"])
        transform = build_transforms(backbone)
    else:
        first_fold = sorted(fold_ckpts.keys())[0]
        model, model_info = _load_model_from_ckpt(fold_ckpts[first_fold], device, args.backbone)
        model_cache[first_fold] = model
        backbone = str(model_info["backbone"])
        label_type = str(model_info["label_type"])
        num_classes = int(model_info["num_classes"])
        transform = build_transforms(backbone)

    feature_group = "gesture_prompt_features" if label_type == "gesture" else "error_prompt_features"
    out_root = args.output_root or os.path.join(args.data_root, feature_group, ckpt_stem)
    print(f"[INFO] Split scheme: {args.split_scheme}")
    print(f"[INFO] Split root: {args.split_root}")
    print(f"[INFO] Label type: {label_type} (num_classes={num_classes})")

    total_videos = 0
    total_frames = 0

    fold_video_map: Dict[str, Dict[str, int]] = {}
    if use_fold_ckpts:
        for task in args.tasks:
            fold_video_map[task] = _load_fold_video_map(args.split_root, task, folds=fold_ids)

    for task in args.tasks:
        vid_frames_dir = os.path.join(args.data_root, "vid_frames", task)
        if not os.path.exists(vid_frames_dir):
            print(f"[WARN] Missing frames dir: {vid_frames_dir}")
            continue

        videos = sorted([d for d in os.listdir(vid_frames_dir) if os.path.isdir(os.path.join(vid_frames_dir, d))])
        print(f"\n[INFO] Task={task} videos={len(videos)}")

        for v in tqdm(videos, desc=f"Extract {task}"):
            out_path = os.path.join(out_root, task, v, "features.pt")
            if os.path.exists(out_path) and not args.force:
                continue
            if use_fold_ckpts:
                video_fold = fold_video_map.get(task, {}).get(v)
                if video_fold is None:
                    print(f"[WARN] No fold assignment for video {v} (task={task}); skipping")
                    continue
                if video_fold not in model_cache:
                    model_cache[video_fold], ckpt_info = _load_model_from_ckpt(
                        fold_ckpts[video_fold],
                        device,
                        backbone_override=backbone,
                    )
                    ckpt_backbone = str(ckpt_info["backbone"])
                    if ckpt_backbone != backbone:
                        raise ValueError(
                            f"Backbone mismatch for fold {video_fold}: expected {backbone}, got {ckpt_backbone}"
                        )
                    if str(ckpt_info["label_type"]) != label_type or int(ckpt_info["num_classes"]) != num_classes:
                        raise ValueError(
                            f"Checkpoint label mismatch for fold {video_fold}: "
                            f"expected ({label_type}, {num_classes}), got ({ckpt_info['label_type']}, {ckpt_info['num_classes']})"
                        )
                    if ckpt_info.get("projected_dim") != model_info.get("projected_dim"):
                        raise ValueError(
                            f"Checkpoint head mismatch for fold {video_fold}: "
                            f"expected projected_dim={model_info.get('projected_dim')}, got {ckpt_info.get('projected_dim')}"
                        )
                model = model_cache[video_fold]
                print(f"[DEBUG] Video={v} task={task} fold={video_fold} ckpt={fold_ckpts[video_fold]}")
            else:
                print(f"[DEBUG] Video={v} task={task} ckpt={fold_ckpts[1]}")
            video_dir = os.path.join(vid_frames_dir, v)
            n = extract_video_features(model, transform, video_dir, out_path, device, batch_size=args.batch_size)
            total_videos += 1
            total_frames += n

    print("\n[INFO] Done.")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] label_type={label_type}")
    print(f"[INFO] backbone={backbone}")
    print(f"[INFO] output_root={out_root}")
    print(f"[INFO] processed_videos={total_videos} processed_frames={total_frames}")


if __name__ == "__main__":
    main()


