"""
Train SAR-RARP50 cross-context model using TWO precomputed feature streams:
- base stream replaces the original `cnn` features:
    - `embed`: existing `embed.npy` (ResNet50)
    - `pt`: `features.pt` extracted by `extract_features_sar_rarp50.py`
      (e.g. CLIP ViT or a SAR-RARP50 error-finetuned ResNet50)
- gesture stream replaces the original `cnnges` features:
    - `ges_embed.npy` or `tri_embed.npy` under each video folder

We keep the SAR_RARP50 train/test split, but create a deterministic validation split from training videos.
"""

import argparse
import copy
import json
import os
import random
import time
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from PIL import Image

try:
    from SAR_RARP.model_gvr import GVRModuleContexPred
    from SAR_RARP.dataloader_rarp50_features import BaseFeatureSpec, SAR_RARP50DualContextDataset, collate_fn_dual_context, split_train_val
    from SAR_RARP.util_rarp50_features import compute_pos_weight_context, eval_context_dual, train_context_dual
except Exception:
    from model_gvr import GVRModuleContexPred
    from dataloader_rarp50_features import BaseFeatureSpec, SAR_RARP50DualContextDataset, collate_fn_dual_context, split_train_val
    from util_rarp50_features import compute_pos_weight_context, eval_context_dual, train_context_dual


def _infer_dims(ds: SAR_RARP50DualContextDataset):
    for i in range(min(len(ds), 200)):
        item = ds[i]
        if item is None:
            continue
        base, ges, labels, _vid = item
        return int(base.size(-1)), int(ges.size(-1))
    raise RuntimeError("Could not infer base/gesture dims (dataset returned only None).")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _attach_context_attention_hooks(model: nn.Module):
    model._attn_trans = []
    model._attn_prompt = []
    hooks = []

    def _trans_hook(_module, _inp, out):
        if isinstance(out, tuple) and len(out) > 1:
            model._attn_trans.append(out[1].detach().cpu())

    def _prompt_hook(_module, _inp, out):
        if isinstance(out, tuple) and len(out) > 1:
            model._attn_prompt.append(out[1].detach().cpu())

    hooks.append(model.trans.register_forward_hook(_trans_hook))
    hooks.append(model.attention.register_forward_hook(_prompt_hook))
    return hooks


def _load_frame(
    data_root: str,
    split_set: str,
    video_name: str,
    logical_frame_idx: int,
    frame_file_stride: int = 12,
    frame_index_base: int = 0,
) -> Optional[Image.Image]:
    frame_idx = int(logical_frame_idx) * int(frame_file_stride) + int(frame_index_base)
    frame_path = os.path.join(data_root, split_set, video_name, "images", f"{frame_idx:09d}.png")
    if not os.path.exists(frame_path):
        return None
    return Image.open(frame_path).convert("RGB")


def _save_attention_figure(
    out_path: str,
    attn_cross: Optional[np.ndarray],
    attn_self: Optional[np.ndarray],
    frame_nums: List[int],
    labels: List[int],
    probs: List[float],
):
    n_rows = 4 if (attn_cross is not None and attn_self is not None) else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(8, len(frame_nums) * 0.25), 9), constrained_layout=True)
    if n_rows == 3:
        axes = np.array(axes).reshape(-1)

    row = 0
    if attn_cross is not None:
        im0 = axes[row].imshow(attn_cross, aspect="auto", interpolation="nearest")
        axes[row].set_title("Gesture-to-Base Attention")
        axes[row].set_ylabel("Gesture Frame")
        axes[row].set_xlabel("Base Frame")
        fig.colorbar(im0, ax=axes[row], fraction=0.02, pad=0.02)
        row += 1

    if attn_self is not None:
        im1 = axes[row].imshow(attn_self, aspect="auto", interpolation="nearest")
        axes[row].set_title("Gesture Self-Attention")
        axes[row].set_ylabel("Query Frame")
        axes[row].set_xlabel("Key Frame")
        fig.colorbar(im1, ax=axes[row], fraction=0.02, pad=0.02)
        row += 1

    axes[row].imshow(np.array([labels]), aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    axes[row].set_ylabel("Error")
    axes[row].set_yticks([])
    axes[row].set_xlabel("Frame Index")
    row += 1

    x = list(range(len(probs))) if probs else [0, 1]
    y = probs if probs else [0.0, 0.0]
    axes[row].plot(x, y, color="black", linewidth=2)
    axes[row].set_ylim(0.0, 1.0)
    axes[row].set_ylabel("Pred Prob")
    axes[row].set_xlabel("Frame Index")
    if frame_nums:
        step = max(1, len(frame_nums) // 10)
        tick_idx = list(range(0, len(frame_nums), step))
        axes[row].set_xticks(tick_idx)
        axes[row].set_xticklabels([str(frame_nums[i]) for i in tick_idx], rotation=45, fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_frames_grid(out_path: str, frames: List[Image.Image], cols: int = 10):
    if not frames:
        return
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    if rows == 1:
        axes = np.array([axes])
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx < len(frames):
            ax.imshow(frames[idx])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def visualize_test_windows(
    model: nn.Module,
    test_ds: SAR_RARP50DualContextDataset,
    args,
    device: torch.device,
    test_split_set: str,
):
    _ensure_dir(args.vis_outdir)
    hooks = _attach_context_attention_hooks(model)
    rng = random.Random(args.vis_seed)

    model.eval()
    saved = 0
    for idx in range(len(test_ds)):
        if rng.random() > args.vis_ratio:
            continue
        if args.vis_max_windows and saved >= args.vis_max_windows:
            break

        vid, start = test_ds.indices[idx]
        item = test_ds[idx]
        if item is None:
            continue
        base, ges, labels, _ = item
        t = int(base.size(0))
        if t <= 0:
            continue

        masks = (torch.arange(t).unsqueeze(0) < torch.tensor([t]).unsqueeze(1)).to(device)
        base = base.unsqueeze(0).to(device)
        ges = ges.unsqueeze(0).to(device)

        model._attn_trans = []
        model._attn_prompt = []
        logits = model(base, ges, masks=masks).squeeze(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)
        frame_nums = [start + i for i in range(t)]

        attn_cross = None
        attn_self = None
        if len(model._attn_trans) > 0:
            attn_cross = model._attn_trans[-1].squeeze(0).numpy()
        if len(model._attn_prompt) > 0:
            attn_self = model._attn_prompt[-1].squeeze(0).numpy()

        base_name = f"{vid}_idx{idx}_s{start}_len{t}"
        attn_path = os.path.join(args.vis_outdir, f"{base_name}_attn.png")
        frames_path = os.path.join(args.vis_outdir, f"{base_name}_frames.png")
        meta_path = os.path.join(args.vis_outdir, f"{base_name}_meta.json")
        _save_attention_figure(attn_path, attn_cross, attn_self, frame_nums, labels_np.tolist(), probs.tolist())

        frames = []
        for fn in frame_nums:
            img = _load_frame(
                args.data_root,
                test_split_set,
                vid,
                fn,
                frame_file_stride=int(args.frame_file_stride),
                frame_index_base=int(args.frame_index_base),
            )
            if img is not None:
                frames.append(img)
        _save_frames_grid(frames_path, frames)

        pred_window = int((probs > 0.5).astype(int).max()) if len(probs) > 0 else 0
        gt_window = int(labels_np.max()) if labels_np.size > 0 else 0
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_id": vid,
                    "start": int(start),
                    "frames": int(t),
                    "gt_label_window": gt_window,
                    "pred_label_window": pred_window,
                },
                f,
                indent=2,
            )
        saved += 1

    for h in hooks:
        h.remove()


def main():
    parser = argparse.ArgumentParser(description="Train SAR_RARP50 cross-context model (dual-feature).")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--base_feature_source", type=str, default="embed", choices=["embed", "pt"])
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="When base_feature_source=pt: model subdir under vid_features (e.g. clip-vit-base-patch32 or a finetuned ResNet50 checkpoint stem)",
    )
    parser.add_argument("--gesture_embed", type=str, default="ges", choices=["ges", "tri"], help="Choose gesture stream: ges_embed or tri_embed")

    parser.add_argument("--segment_length", type=int, default=40)
    parser.add_argument("--train_step_size", type=int, default=6)
    parser.add_argument("--test_step_size", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (0 disables multiprocessing)")
    parser.add_argument("--label_mode", type=str, default="segment", choices=["segment", "frame"], help="segment: window label repeated per frame; frame: use raw error_GT per-frame labels")

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_embed", type=int, default=0, help="Embedding dim inside model (0 => use gesture dim)")

    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--layer_norm1", type=int, default=0)
    parser.add_argument("--layer_norm2", type=int, default=1)
    parser.add_argument("--layer_norm3", type=int, default=0)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--use_test_as_val", type=int, default=0, help="If 1, skip val split and use test set for val metrics")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed (also controls the train/val video split). Unset leaves RNGs unseeded.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_pos_weight", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize_test", type=int, default=0, help="If 1, save test attention/labels/frame visualizations")
    parser.add_argument("--vis_ratio", type=float, default=0.3, help="Fraction of test windows to visualize")
    parser.add_argument("--vis_outdir", type=str, default="./outputs/sar_rarp50/gvrcross_viz", help="Output directory for visualizations")
    parser.add_argument("--vis_seed", type=int, default=None, help="Optional random seed for visualization sampling")
    parser.add_argument("--vis_max_windows", type=int, default=0, help="Max windows to visualize (0 = no cap)")
    parser.add_argument("--frame_file_stride", type=int, default=12, help="Frame filename stride from logical index (e.g., 12 for 60Hz->5Hz)")
    parser.add_argument("--frame_index_base", type=int, default=0, help="Optional filename offset added after stride mapping")
    args = parser.parse_args()

    args.position = True if args.position == 1 else False
    args.layer_norm1 = True if args.layer_norm1 == 1 else False
    args.layer_norm2 = True if args.layer_norm2 == 1 else False
    args.layer_norm3 = True if args.layer_norm3 == 1 else False
    args.use_pos_weight = True if args.use_pos_weight == 1 else False
    args.visualize_test = True if args.visualize_test == 1 else False
    args.use_test_as_val = True if args.use_test_as_val == 1 else False

    device = torch.device(args.device)
    train_pkl_dir = os.path.join(args.data_root, "train_emb_DINOv2")
    test_pkl_dir = os.path.join(args.data_root, "test_emb_DINOv2")
    train_split_set = "training_set"
    test_split_set = "testing_set"

    if args.base_feature_source == "embed":
        base = BaseFeatureSpec(source="embed", model=None)
        base_tag = "embed"
    else:
        if not args.base_model:
            raise ValueError("--base_model is required when --base_feature_source=pt")
        base = BaseFeatureSpec(source="pt", model=args.base_model)
        base_tag = f"pt_{args.base_model}"

    train_video_ids = sorted([f.replace(".pkl", "") for f in os.listdir(train_pkl_dir) if f.endswith(".pkl")])
    if args.use_test_as_val:
        tr_ids = train_video_ids
        va_ids = []
        print(
            f"[INFO] Train videos: {len(tr_ids)} (all training set) | Val videos: (using test set) | "
            f"Test videos: {len(os.listdir(test_pkl_dir))}"
        )
    else:
        tr_ids, va_ids = split_train_val(train_video_ids, val_ratio=args.val_ratio, seed=args.seed)
        print(f"[INFO] Train videos: {len(tr_ids)} | Val videos: {len(va_ids)} | Test videos: {len(os.listdir(test_pkl_dir))}")

    train_ds = SAR_RARP50DualContextDataset(
        data_root=args.data_root,
        pkl_dir=train_pkl_dir,
        split_set=train_split_set,
        base=base,
        gesture_embed=args.gesture_embed,
        video_ids=tr_ids,
        segment_length=args.segment_length,
        step_size=args.train_step_size,
        label_mode=args.label_mode,
    )
    if args.use_test_as_val:
        val_ds = SAR_RARP50DualContextDataset(
            data_root=args.data_root,
            pkl_dir=test_pkl_dir,
            split_set=test_split_set,
            base=base,
            gesture_embed=args.gesture_embed,
            video_ids=None,
            segment_length=args.segment_length,
            step_size=args.test_step_size,
            label_mode=args.label_mode,
        )
    else:
        val_ds = SAR_RARP50DualContextDataset(
            data_root=args.data_root,
            pkl_dir=train_pkl_dir,
            split_set=train_split_set,
            base=base,
            gesture_embed=args.gesture_embed,
            video_ids=va_ids,
            segment_length=args.segment_length,
            step_size=args.test_step_size,
            label_mode=args.label_mode,
        )
    test_ds = SAR_RARP50DualContextDataset(
        data_root=args.data_root,
        pkl_dir=test_pkl_dir,
        split_set=test_split_set,
        base=base,
        gesture_embed=args.gesture_embed,
        video_ids=None,
        segment_length=args.segment_length,
        step_size=args.test_step_size,
        label_mode=args.label_mode,
    )

    dl_kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": (device.type == "cuda"),
    }
    if int(args.num_workers) > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dual_context, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_dual_context, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_dual_context, **dl_kwargs)

    base_dim, ges_dim = _infer_dims(train_ds)
    d_embed = int(args.d_embed) if int(args.d_embed) > 0 else int(ges_dim)
    print(f"[INFO] base={base_tag} base_dim={base_dim} gesture={args.gesture_embed} ges_dim={ges_dim} d_embed={d_embed}")

    model = GVRModuleContexPred(
        base_dim=base_dim,
        ges_dim=ges_dim,
        d_embed=d_embed,
        num_heads=args.num_heads,
        n_frames=args.segment_length,
        layer_norm1=args.layer_norm1,
        layer_norm2=args.layer_norm2,
        layer_norm3=args.layer_norm3,
        position=args.position,
        dropout=args.dropout,
    ).to(device)

    if args.use_pos_weight:
        _, criterion = compute_pos_weight_context(train_loader, device)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)

    out_dir = "./outputs/sar_rarp50/gvrcrosscontext_features"
    os.makedirs(out_dir, exist_ok=True)
    run_tag = (
        f"base{base_tag}_ges{args.gesture_embed}_lr{args.learning_rate}_wd{args.weight_decay}_"
        f"bs{args.batch_size}_seed{args.seed}_trstep{args.train_step_size}_testep{args.test_step_size}"
    )
    csv_path = os.path.join(out_dir, f"gvrcross_{run_tag}_metrics.csv")
    ckpt_path = os.path.join(out_dir, f"gvrcross_{run_tag}_best.pth")

    best_val_f1 = 0.0
    best_state = None
    best_epoch = None
    epochs_wo = 0
    logs = []

    start_time = time.time()
    for epoch in range(args.num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch+1}/{args.num_epochs} (LR={lr:.6f})")

        tr_f1, tr_acc, tr_j, tr_auc, tr_loss = train_context_dual(
            model, train_loader, criterion, optimizer, device, max_grad_norm=args.max_grad_norm
        )
        va_f1, va_acc, va_j, va_auc = eval_context_dual(model, val_loader, device)
        te_f1, te_acc, te_j, te_auc = eval_context_dual(model, test_loader, device)

        scheduler.step(va_f1)

        logs.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": tr_loss,
                "train_f1": tr_f1,
                "train_acc": tr_acc,
                "train_jaccard": tr_j,
                "train_auc": tr_auc,
                "val_f1": va_f1,
                "val_acc": va_acc,
                "val_jaccard": va_j,
                "val_auc": va_auc,
                "test_f1": te_f1,
                "test_acc": te_acc,
                "test_jaccard": te_j,
                "test_auc": te_auc,
            }
        )

        print(
            f"Train: loss={tr_loss:.4f} f1={tr_f1:.4f} bacc={tr_acc:.4f} j={tr_j:.4f} auc={tr_auc:.4f}"
        )
        print(f"Val:                f1={va_f1:.4f} bacc={va_acc:.4f} j={va_j:.4f} auc={va_auc:.4f}")
        print(f"Test:               f1={te_f1:.4f} bacc={te_acc:.4f} j={te_j:.4f} auc={te_auc:.4f}")

        if va_f1 > best_val_f1:
            best_val_f1 = float(va_f1)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_wo = 0
            print(f"  *** New best val_f1={best_val_f1:.4f} (epoch {best_epoch}) ***")
        else:
            epochs_wo += 1
            if epochs_wo >= args.patience:
                print("  Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "base_feature_source": args.base_feature_source,
            "base_model": args.base_model,
            "gesture_embed": args.gesture_embed,
            "base_dim": base_dim,
            "ges_dim": ges_dim,
            "d_embed": d_embed,
            "args": vars(args),
        },
        ckpt_path,
    )
    pd.DataFrame(logs).to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved best checkpoint: {ckpt_path}")
    print(f"[INFO] Saved logs: {csv_path}")
    print(f"[INFO] Total time: {(time.time() - start_time)/3600:.2f} hours")

    if args.visualize_test:
        visualize_test_windows(model, test_ds, args, device, test_split_set)


if __name__ == "__main__":
    main()

