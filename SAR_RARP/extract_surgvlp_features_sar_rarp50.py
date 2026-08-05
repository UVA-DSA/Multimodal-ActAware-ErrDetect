#!/usr/bin/env python3
"""
Extract SurgVLP image frame features and SurgVLP-encoded prompt embeddings for SAR_RARP50.

Inputs:
  - Frames:
      data/SAR_RARP50/{training_set,testing_set}/{video_id}/images/*.png
  - Prompt text lists:
      SAR_RARP/textprompt.py  (error/gesture/context/gesture_error/lowlevel_gesture_error)

Outputs:
  - Per-frame image embeddings (dict[int -> tensor(D)]):
      data/SAR_RARP50/vid_features/surgvlp/{training_set,testing_set}/{video_id}/features.pt

  - Prompt embeddings for ALL prompt types (single file):
      data/SAR_RARP50/prompt_features_surgvlp/prompts.pt
    Format: {
      "meta": {...},
      "prompt_types": {
         "<prompt_type>": {"prompts": [str,...], "text_emb": Tensor(J,D)}
      }
    }

Notes:
  - This script uses the SurgVLP package vendored under ./SurgVLP.
  - It does NOT require SAR_RARP50 .pkl files; it only needs frames for image features.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm


PROMPT_TYPES = ["error", "gesture", "context", "gesture_error", "lowlevel_gesture_error"]
_FRAME_RE = re.compile(r"(\d+)\.png$")


def _ensure_surgvlp_on_path(repo_root: str):
    surgvlp_root = os.path.join(repo_root, "SurgVLP")
    if os.path.isdir(surgvlp_root) and surgvlp_root not in sys.path:
        sys.path.insert(0, surgvlp_root)


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
def _extract_video_features(model, preprocess, images_dir: str, device: str, batch_size: int) -> Dict[int, torch.Tensor]:
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
                img = preprocess(Image.open(fp)).to(torch.float32)
                imgs.append(img)
                idxs.append(frame_idx)
            except Exception as e:
                print(f"[WARN] Failed to read {fp}: {e}")
        if not imgs:
            continue
        x = torch.stack(imgs).to(device)  # (B,C,H,W)
        out = model(x, None, mode="video")  # dict with img_emb
        emb = out["img_emb"].to(torch.float32)  # (B,D)
        for j, frame_idx in enumerate(idxs):
            feats[frame_idx] = emb[j].detach().cpu()

    return feats


@torch.no_grad()
def _encode_prompts(model, surgvlp_tokenize, prompts: List[str], device: str) -> torch.Tensor:
    tokens = surgvlp_tokenize(prompts, device=device)
    out = model(None, tokens, mode="text")
    emb = out["text_emb"].to(torch.float32)  # (J,D)
    return emb.detach().cpu()


def main():
    p = argparse.ArgumentParser(description="Extract SurgVLP features/prompts for SAR_RARP50")
    p.add_argument("--repo_root", type=str, default=".", help="Repo root containing SurgVLP/ and SAR_RARP/")
    p.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    p.add_argument("--splits", type=str, nargs="+", default=["training_set", "testing_set"], choices=["training_set", "testing_set"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--max_videos", type=int, default=0, help="If >0, process at most this many videos per split (debug)")
    p.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Optional path to SurgVLP weights .pth (if omitted, SurgVLP will download)",
    )
    p.add_argument(
        "--download_root",
        type=str,
        default=None,
        help="Optional download cache root for SurgVLP weights",
    )
    args = p.parse_args()

    _ensure_surgvlp_on_path(os.path.abspath(args.repo_root))

    try:
        import surgvlp  # type: ignore
        from SurgVLP.tests.config_surgvlp import config as _cfg  # type: ignore
    except Exception as e:
        raise SystemExit(
            "[ERROR] Could not import SurgVLP. Ensure dependencies are installed and repo_root is correct.\n"
            f"Original error: {e}"
        )

    # Build model config (from SurgVLP's provided test config)
    model_config = _cfg["model_config"]
    model, preprocess = surgvlp.load(model_config, device=args.device, download_root=args.download_root, pretrain=args.pretrain)
    model = model.to(args.device).eval()

    # Load prompt lists from SAR_RARP/textprompt.py
    sys.path.insert(0, os.path.abspath(args.repo_root))
    from SAR_RARP import textprompt as tp  # type: ignore

    prompt_sets = {
        "error": list(tp.error_prompts),
        "gesture": list(tp.gesture_prompts),
        "context": list(tp.context_promtps),
        "gesture_error": list(tp.gesture_error_prompt),
        "lowlevel_gesture_error": list(tp.lowlevelges_error_prompt),
    }

    # 1) Encode and save all prompt embeddings
    prompt_out_dir = os.path.join(args.data_root, "prompt_features_surgvlp")
    os.makedirs(prompt_out_dir, exist_ok=True)
    prompt_out_path = os.path.join(prompt_out_dir, "prompts.pt")
    if args.force or (not os.path.exists(prompt_out_path)):
        prompt_blob = {"meta": {"model": "SurgVLP"}, "prompt_types": {}}
        for pt_name in PROMPT_TYPES:
            prompts = prompt_sets[pt_name]
            text_emb = _encode_prompts(model, surgvlp.tokenize, prompts, device=args.device)
            prompt_blob["prompt_types"][pt_name] = {"prompts": prompts, "text_emb": text_emb}
            print(f"[INFO] Encoded prompts: {pt_name} (J={len(prompts)} D={tuple(text_emb.shape)})")
        torch.save(prompt_blob, prompt_out_path)
        print(f"[INFO] Saved prompt embeddings: {prompt_out_path}")
    else:
        print(f"[INFO] Prompt embeddings exist, skipping: {prompt_out_path} (use --force to overwrite)")

    # 2) Extract per-frame image features
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

        for vid in tqdm(video_dirs, desc=f"Extracting SurgVLP {split}"):
            images_dir = os.path.join(split_dir, vid, "images")
            if not os.path.isdir(images_dir):
                continue
            out_path = os.path.join(args.data_root, "vid_features", "surgvlp", split, vid, "features.pt")
            if os.path.exists(out_path) and not args.force:
                continue

            feats = _extract_video_features(model, preprocess, images_dir, args.device, args.batch_size)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(feats, out_path)
            total_videos += 1
            total_frames += len(feats)

    print("\n[INFO] Done.")
    print(f"[INFO] Processed videos={total_videos} frames={total_frames}")
    print(f"[INFO] Feature root: {os.path.join(args.data_root, 'vid_features', 'surgvlp')}")


if __name__ == "__main__":
    main()


