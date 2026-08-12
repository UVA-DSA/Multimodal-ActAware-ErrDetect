#!/usr/bin/env python3
"""
Per-window inference latency of our models vs CoG's GVR module.

All models are profiled under identical conditions: the same window length, the
same pre-extracted 2048-d ResNet50 features, the same prompt set (so the number of
prompt tokens is held constant), batch size 1 (streaming / real-time inference),
and CUDA synchronisation around every timed call.

Reported latency is **model inference only** — it excludes frame decoding, resizing
and the frozen ResNet/CLIP encoders, which are shared by all of these models and
dominate the end-to-end budget.

  python tools/profiling/profile_window_latency.py --window 10 --iters 200
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "jigsaws")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_gvr_features import GVRModulePredFeatures, GVRModulePredKinFeatures  # noqa: E402
from model_cog_gvr_window import CoGGVRWindowFeatures  # noqa: E402
from textprompts import gesture_error_prompt  # noqa: E402


def count_params(model: torch.nn.Module) -> int:
    """Trainable + buffered parameters actually executed at inference."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def time_model(fn: Callable[[], torch.Tensor], device: torch.device, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10, help="Window length in samples (10 = 2 s at 5 Hz)")
    ap.add_argument("--feature_dim", type=int, default=2048, help="Frame feature dim (ResNet50 = 2048)")
    ap.add_argument("--batch_size", type=int, default=1, help="1 = streaming inference")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--preset", choices=["tuned", "paper"], default="tuned",
                    help="tuned = the shipped configuration; paper = the architecture profiled in "
                         "the supplementary (TCN pooling, concat fusion, attention temperature 1.0)")
    ap.add_argument("--d_cog", type=int, default=64, help="CoG transformer width (its --dmodel)")
    ap.add_argument("--cog_len_q", type=int, default=None, help="CoG causal history; defaults to the window length")
    ap.add_argument("--include_cog_full", action="store_true",
                    help="Also profile the complete CoG model (GVR + multi-scale temporal reasoning), "
                         "built with the same spec the supplementary profiler used.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join("results", "LATENCY_PROFILE.csv"))
    args = ap.parse_args()

    device = torch.device(args.device)
    prompts = gesture_error_prompt
    B, T, D = args.batch_size, args.window, args.feature_dim

    feats = torch.randn(B, T, D, device=device)
    kin = torch.randn(B, T, 14, device=device)
    masks = torch.ones(B, T, dtype=torch.bool, device=device)

    print(f"[cfg] device={device} window={T} feature_dim={D} batch={B} prompts={len(prompts)} "
          f"warmup={args.warmup} iters={args.iters}")
    if device.type == "cuda":
        print(f"[cfg] gpu={torch.cuda.get_device_name(0)}")

    # Prompt embeddings are computed once and shared, so no model pays a CLIP cost here
    # (the text encoder runs offline; prompts are fixed at inference time).
    shared_prompts = CoGGVRWindowFeatures._clip_prompt_features(prompts)

    if args.preset == "paper":
        prompt_cfg = dict(pooled_mode="tcn", fusion="concat", use_pooled_branch=True, attn_temperature=1.0)
        kin_cfg = dict(pooled_mode="tcn", fusion="concat", use_pooled_branch=False, attn_temperature=1.0)
    else:
        prompt_cfg = dict(pooled_mode="mean", fusion="concat", use_pooled_branch=True, attn_temperature=0.7)
        kin_cfg = dict(pooled_mode="mean", fusion="gated", use_pooled_branch=False, attn_temperature=1.0)
    print(f"[cfg] preset={args.preset} prompt={prompt_cfg} kin={kin_cfg}")

    ours_prompt = GVRModulePredFeatures(
        gesture_prompts=prompts, d_text=512, d_model=D, num_heads=args.num_heads,
        segment_length=T, position=True, dropout=0.3, prompt_features=shared_prompts,
        **prompt_cfg,
    ).to(device).eval()

    ours_kin = GVRModulePredKinFeatures(
        gesture_prompts=prompts, d_text=512, d_model=D, num_heads=args.num_heads,
        segment_length=T, position=True, dropout=0.5, prompt_features=shared_prompts,
        kin_fusion="concat", kin_dim_ratio=8, **kin_cfg,
    ).to(device).eval()

    cog_gvr = CoGGVRWindowFeatures(
        gesture_prompts=prompts, d_text=512, d_model=D, d_cog=args.d_cog,
        segment_length=T, len_q=args.cog_len_q, dropout=0.1,
        prompt_features=shared_prompts, device=str(device),
    ).to(device).eval()

    entries = [
        ("Ours - Activity Prompting (Img+Txt)", ours_prompt, lambda: ours_prompt(feats, masks=masks)),
        ("Ours - Activity Kin. Fusion (Img+Txt+Kin)", ours_kin, lambda: ours_kin(feats, kin, masks=masks)),
        ("CoG GVR module (window-level)", cog_gvr, lambda: cog_gvr(feats, masks=masks)),
    ]

    if args.include_cog_full:
        # Same construction as the supplementary profiler's build_cog_spec().
        import tempfile
        from types import SimpleNamespace
        cog_dir = os.path.join(REPO_ROOT, "baselines", "Chain-of-Gesture")
        if cog_dir not in sys.path:
            sys.path.insert(0, cog_dir)
        import models as cog_module

        prompt_file = tempfile.NamedTemporaryFile(prefix="cog_prompt_", suffix=".pt", delete=False).name
        cog_args = SimpleNamespace(train=1, k=T, layers=10, stages=8, lambda_=0.15, dmodel=args.d_cog, len_q=T)
        cog_full = cog_module.COG(
            cog_args, num_layers_Basic=11, num_layers_R=cog_args.layers, num_R=3,
            num_f_maps=64, num_f_dim=D, num_classes=2, causal_conv=True,
            d_model=cog_args.dmodel, d_q=cog_args.dmodel // 8, len_q=cog_args.len_q,
            device=device, gest_prompt=prompt_file,
        ).to(device).eval()
        entries.append(("CoG full (GVR + multi-scale temporal reasoning)", cog_full,
                        lambda: cog_full(feats)[0][0]))

    rows = []
    for name, model, fn in entries:
        stats = time_model(fn, device, args.warmup, args.iters)
        params = count_params(model)
        rows.append({"model": name, "params_M": params / 1e6, **stats})
        print(f"  {name:44s} {stats['mean_ms']:7.3f} ± {stats['std_ms']:.3f} ms  "
              f"(median {stats['median_ms']:.3f}, p95 {stats['p95_ms']:.3f})  {params/1e6:.2f} M params")

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df["window"] = T
        df["batch_size"] = B
        df["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
