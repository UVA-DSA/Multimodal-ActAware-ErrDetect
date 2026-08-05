"""
Train CoOp-style prompt context embeddings for SAR-RARP50 gesture recognition.

This script learns trainable prompt context tokens (pre and post) using CLIP
image-text contrastive classification (frame -> gesture class), then exports
prompt embeddings for a user-provided partial-prompt dictionary.

Dataset layout:
  Frames:
    data/SAR_RARP50/{training_set,testing_set}/video_xx/images/*
  Gesture labels:
    data/SAR_RARP50/gestures/{train,test}/video_xx/action_discrete.txt

The annotation file format is:
  frame_id,gesture_id
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


# ---------------------------------------------------------------------------
# Placeholder mapping.
# Fill this with your own mapping from partial prompt text -> gesture id(s).
# Example:
# {
#   "A surgeon is picking up the needle": [1],
#   "A surgeon is tying a knot": [5],
#   "The surgeon is moving instruments carefully": [1, 2, 3],
# }
# ---------------------------------------------------------------------------
PARTIAL_PROMPT_TO_GESTURE_IDS: Dict[str, List[int]] = {
    "performing another action": [0],                    # G0
    "picking up the needle": [1],                        # G1
    "positioning the needle tip": [2],                   # G2
    "pushing the needle through the tissue": [3],        # G3
    "pulling the needle out of the tissue": [4],         # G4
    "tying a knot": [5],                                 # G5
    "cutting the suture": [6],                           # G6
    "returning/dropping the needle": [7]                # G7
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_annotation(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            frame_id = int(parts[0].strip())
            gesture_id = int(parts[1].strip())
            out[frame_id] = gesture_id
    return out


def extract_frame_number(filename: str) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1


class SARRARPFrameGestureDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        frame_split: str,
        gesture_split: str,
        allowed_gesture_ids: Sequence[int],
    ):
        self.samples: List[Tuple[str, int]] = []
        allowed = set(int(x) for x in allowed_gesture_ids)

        frame_root = os.path.join(data_root, frame_split)
        gesture_root = os.path.join(data_root, "gestures", gesture_split)
        if not os.path.isdir(frame_root):
            raise FileNotFoundError(f"Frame split path not found: {frame_root}")
        if not os.path.isdir(gesture_root):
            raise FileNotFoundError(f"Gesture split path not found: {gesture_root}")

        video_dirs = sorted(
            [d for d in os.listdir(frame_root) if d.startswith("video_") and os.path.isdir(os.path.join(frame_root, d))]
        )
        for vid in video_dirs:
            ann_path = os.path.join(gesture_root, vid, "action_discrete.txt")
            img_dir = os.path.join(frame_root, vid, "images")
            if not os.path.isfile(ann_path) or not os.path.isdir(img_dir):
                continue

            frame_to_gesture = parse_annotation(ann_path)
            img_files = sorted(
                [x for x in os.listdir(img_dir) if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
            )
            for img_name in img_files:
                frame_num = extract_frame_number(img_name)
                if frame_num < 0:
                    continue
                gesture_id = frame_to_gesture.get(frame_num)
                if gesture_id is None or gesture_id not in allowed:
                    continue
                self.samples.append((os.path.join(img_dir, img_name), int(gesture_id)))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for frame_split={frame_split}, gesture_split={gesture_split}. "
                "Check PARTIAL_PROMPT_TO_GESTURE_IDS and dataset paths."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, y


class CoOpPromptLearner(nn.Module):
    def __init__(
        self,
        clip_model: CLIPModel,
        tokenizer: CLIPTokenizer,
        n_pre_ctx: int,
        n_post_ctx: int,
        max_text_len: int = 77,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.n_pre_ctx = int(n_pre_ctx)
        self.n_post_ctx = int(n_post_ctx)
        self.max_text_len = int(max_text_len)

        text_hidden = int(clip_model.text_model.config.hidden_size)
        self.pre_ctx = nn.Parameter(torch.empty(self.n_pre_ctx, text_hidden))
        self.post_ctx = nn.Parameter(torch.empty(self.n_post_ctx, text_hidden))
        nn.init.normal_(self.pre_ctx, std=0.02)
        nn.init.normal_(self.post_ctx, std=0.02)

    def trainable_parameters(self):
        return [self.pre_ctx, self.post_ctx]

    def _build_text_inputs(
        self, prompt_texts: Sequence[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(list(prompt_texts), add_special_tokens=False, return_attention_mask=False)
        token_lists = tok["input_ids"]

        bos_id = int(self.tokenizer.bos_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        pad_id = int(self.tokenizer.pad_token_id)

        token_embedding = self.clip_model.text_model.embeddings.token_embedding
        bos_embed = token_embedding(torch.tensor([bos_id], device=device)).squeeze(0)
        eos_embed = token_embedding(torch.tensor([eos_id], device=device)).squeeze(0)
        pad_embed = token_embedding(torch.tensor([pad_id], device=device)).squeeze(0)

        embeds_all: List[torch.Tensor] = []
        attn_all: List[torch.Tensor] = []
        eos_pos_all: List[int] = []

        for class_token_ids in token_lists:
            class_token_ids = list(class_token_ids)
            reserved = 1 + self.n_pre_ctx + self.n_post_ctx + 1  # BOS + pre + post + EOS
            max_class_tokens = max(1, self.max_text_len - reserved)
            class_token_ids = class_token_ids[:max_class_tokens]

            if len(class_token_ids) > 0:
                class_ids_t = torch.tensor(class_token_ids, dtype=torch.long, device=device)
                class_emb = token_embedding(class_ids_t)
            else:
                class_emb = torch.empty(0, bos_embed.numel(), device=device)

            seq = torch.cat(
                [
                    bos_embed.unsqueeze(0),
                    self.pre_ctx,
                    class_emb,
                    self.post_ctx,
                    eos_embed.unsqueeze(0),
                ],
                dim=0,
            )
            seq_len = int(seq.size(0))
            eos_pos = seq_len - 1
            eos_pos_all.append(eos_pos)

            if seq_len < self.max_text_len:
                pad_len = self.max_text_len - seq_len
                seq = torch.cat([seq, pad_embed.unsqueeze(0).expand(pad_len, -1)], dim=0)
                attn = torch.cat(
                    [
                        torch.ones(seq_len, dtype=torch.long, device=device),
                        torch.zeros(pad_len, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
            else:
                attn = torch.ones(self.max_text_len, dtype=torch.long, device=device)
            embeds_all.append(seq)
            attn_all.append(attn)

        inputs_embeds = torch.stack(embeds_all, dim=0)      # (C, L, H)
        attention_mask = torch.stack(attn_all, dim=0)       # (C, L)
        eos_positions = torch.tensor(eos_pos_all, dtype=torch.long, device=device)  # (C,)
        return inputs_embeds, attention_mask, eos_positions

    @staticmethod
    def _causal_4d_mask(batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # Additive causal mask: 0 for allowed positions, -inf for disallowed future positions.
        mask = torch.full((seq_len, seq_len), fill_value=torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    def encode_texts(self, prompt_texts: Sequence[str]) -> torch.Tensor:
        device = self.pre_ctx.device
        inputs_embeds, _attention_mask, eos_positions = self._build_text_inputs(prompt_texts, device=device)
        bs, seq_len, _ = inputs_embeds.shape

        # In this transformers build, CLIPModel.text_model is already CLIPTextTransformer.
        text_tr = self.clip_model.text_model
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
        pos_emb = text_tr.embeddings.position_embedding(pos_ids)
        hidden_states = inputs_embeds + pos_emb

        causal_attention_mask = self._causal_4d_mask(
            batch_size=bs,
            seq_len=seq_len,
            dtype=hidden_states.dtype,
            device=device,
        )
        encoder_outputs = text_tr.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        last_hidden = text_tr.final_layer_norm(encoder_outputs.last_hidden_state)  # (C, L, H)
        idx = torch.arange(last_hidden.size(0), device=device)
        eos_hidden = last_hidden[idx, eos_positions]  # (C, H)
        text_features = self.clip_model.text_projection(eos_hidden)  # (C, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return text_features


def build_gesture_to_prompts(partial_prompt_to_gesture_ids: Dict[str, List[int]]) -> Dict[int, List[str]]:
    g2p: Dict[int, List[str]] = defaultdict(list)
    for prompt, gids in partial_prompt_to_gesture_ids.items():
        for gid in gids:
            g2p[int(gid)].append(prompt)
    return dict(g2p)


def collate_images(batch, processor: CLIPImageProcessor, gesture_to_idx: Dict[int, int]):
    images, labels = zip(*batch)
    pixel_values = processor(images=list(images), return_tensors="pt")["pixel_values"]
    y = torch.tensor([gesture_to_idx[int(v)] for v in labels], dtype=torch.long)
    return pixel_values, y


def _plot_training_curves(history: List[Dict[str, float]], out_png: str) -> None:
    if len(history) == 0:
        return
    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    train_acc = [float(h["train_acc"]) for h in history]
    test_acc = [float(h["test_acc"]) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, test_acc, marker="o", label="test_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _log(msg: str, use_tqdm: bool) -> None:
    if use_tqdm and tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


@torch.no_grad()
def evaluate(
    clip_model: CLIPModel,
    coop: CoOpPromptLearner,
    loader: DataLoader,
    class_texts: Sequence[str],
    device: torch.device,
) -> float:
    coop.eval()
    clip_model.eval()
    total = 0
    correct = 0
    for pixel_values, y in loader:
        pixel_values = pixel_values.to(device)
        y = y.to(device)

        image_features = clip_model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        text_features = coop.encode_texts(class_texts)

        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (B,C)
        pred = logits.argmax(dim=1)
        total += int(y.numel())
        correct += int((pred == y).sum().item())
    return (correct / total) if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train CoOp prompt embeddings on SAR-RARP50 gestures.")
    parser.add_argument("--data_root", type=str, default="./data/SAR_RARP50")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--n_pre_ctx", type=int, default=4, help="Number of trainable context tokens before prompt text")
    parser.add_argument("--n_post_ctx", type=int, default=4, help="Number of trainable context tokens after prompt text")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed. Unset leaves RNGs unseeded.")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Optional cap on train samples (0=no cap)")
    parser.add_argument("--report_every", type=int, default=100, help="Show running metrics every N training batches")
    parser.add_argument("--show_progress", type=int, default=1, help="Use tqdm progress bars if available")
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/SAR_RARP50/prompt_features_coop/prompts.pt",
        help="Output .pt file storing prompt texts and learned embeddings",
    )
    args = parser.parse_args()

    if len(PARTIAL_PROMPT_TO_GESTURE_IDS) == 0:
        raise ValueError(
            "Please fill PARTIAL_PROMPT_TO_GESTURE_IDS in this script before training."
        )

    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    partial_prompt_texts = list(PARTIAL_PROMPT_TO_GESTURE_IDS.keys())
    gesture_to_prompts = build_gesture_to_prompts(PARTIAL_PROMPT_TO_GESTURE_IDS)
    gesture_ids = sorted(gesture_to_prompts.keys())
    gesture_to_idx = {g: i for i, g in enumerate(gesture_ids)}
    idx_to_gesture = {i: g for g, i in gesture_to_idx.items()}
    class_texts = [" ".join(gesture_to_prompts[g]) for g in gesture_ids]

    train_ds = SARRARPFrameGestureDataset(
        data_root=args.data_root,
        frame_split="training_set",
        gesture_split="train",
        allowed_gesture_ids=gesture_ids,
    )
    test_ds = SARRARPFrameGestureDataset(
        data_root=args.data_root,
        frame_split="testing_set",
        gesture_split="test",
        allowed_gesture_ids=gesture_ids,
    )

    if args.max_train_samples > 0 and args.max_train_samples < len(train_ds):
        keep_idx = np.random.RandomState(args.seed).choice(len(train_ds), size=args.max_train_samples, replace=False)
        keep_idx = sorted(int(i) for i in keep_idx.tolist())
        train_ds.samples = [train_ds.samples[i] for i in keep_idx]

    print(f"[INFO] Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    print(f"[INFO] Gesture IDs used: {gesture_ids}")

    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model_name)
    processor = CLIPImageProcessor.from_pretrained(args.clip_model_name)

    # Freeze CLIP backbone; only train context tokens.
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    coop = CoOpPromptLearner(
        clip_model=clip_model,
        tokenizer=tokenizer,
        n_pre_ctx=args.n_pre_ctx,
        n_post_ctx=args.n_post_ctx,
        max_text_len=int(clip_model.text_model.config.max_position_embeddings),
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_images(b, processor, gesture_to_idx),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_images(b, processor, gesture_to_idx),
    )

    optimizer = optim.AdamW(coop.trainable_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    history: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        coop.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        batches = train_loader
        use_tqdm = bool(args.show_progress == 1 and tqdm is not None)
        _log(f"[INFO] Epoch {epoch+1:03d}/{args.epochs} started", use_tqdm=use_tqdm)
        if use_tqdm:
            batches = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:03d}/{args.epochs}",
                leave=True,
                dynamic_ncols=True,
            )

        for step_idx, (pixel_values, y) in enumerate(batches, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                image_features = clip_model.get_image_features(pixel_values=pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            text_features = coop.encode_texts(class_texts)
            logit_scale = clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(y.size(0))
            running_loss += float(loss.item()) * bs
            running_correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += bs

            if use_tqdm:
                batches.set_postfix(
                    {
                        "loss": f"{(running_loss / max(seen, 1)):.4f}",
                        "acc": f"{(running_correct / max(seen, 1)):.4f}",
                    }
                )
            elif args.report_every > 0 and (step_idx % args.report_every == 0):
                _log(
                    f"  [train] epoch={epoch+1:03d} step={step_idx:04d} "
                    f"loss={running_loss / max(seen, 1):.5f} acc={running_correct / max(seen, 1):.5f}",
                    use_tqdm=use_tqdm,
                )

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        test_acc = evaluate(clip_model, coop, test_loader, class_texts, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
            }
        )
        _log(
            f"Epoch {epoch+1:03d}: train_loss={train_loss:.5f} "
            f"train_acc={train_acc:.5f} test_acc={test_acc:.5f}",
            use_tqdm=use_tqdm,
        )

        if test_acc > best_acc:
            best_acc = float(test_acc)
            best_state = {
                "pre_ctx": coop.pre_ctx.detach().cpu(),
                "post_ctx": coop.post_ctx.detach().cpu(),
            }

    if best_state is None:
        raise RuntimeError("No model state captured during training.")

    with torch.no_grad():
        coop.pre_ctx.copy_(best_state["pre_ctx"].to(device))
        coop.post_ctx.copy_(best_state["post_ctx"].to(device))
        prompt_embeddings = coop.encode_texts(partial_prompt_texts).detach().cpu()

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report_dir = os.path.join(out_dir if out_dir else ".", "coop_reports")
    os.makedirs(report_dir, exist_ok=True)

    torch.save(
        {
            "clip_model_name": args.clip_model_name,
            "n_pre_ctx": int(args.n_pre_ctx),
            "n_post_ctx": int(args.n_post_ctx),
            "partial_prompt_to_gesture_ids": PARTIAL_PROMPT_TO_GESTURE_IDS,
            "prompt_texts": partial_prompt_texts,                # order-preserving list
            "prompt_embeddings": prompt_embeddings,              # (num_prompts, text_dim)
            "class_gesture_ids": gesture_ids,                    # class order used in training
            "class_texts": class_texts,
            "best_test_acc": float(best_acc),
            "seed": -1 if args.seed is None else int(args.seed),
            "history": history,
        },
        args.out_path,
    )

    meta_path = args.out_path + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_model_name": args.clip_model_name,
                "n_pre_ctx": int(args.n_pre_ctx),
                "n_post_ctx": int(args.n_post_ctx),
                "num_partial_prompts": len(partial_prompt_texts),
                "num_gesture_classes": len(gesture_ids),
                "best_test_acc": float(best_acc),
                "report_dir": report_dir,
            },
            f,
            indent=2,
        )

    history_csv = os.path.join(report_dir, "training_history.csv")
    history_json = os.path.join(report_dir, "training_history.json")
    curve_png = os.path.join(report_dir, "training_curves.png")
    pd.DataFrame(history).to_csv(history_csv, index=False)
    with open(history_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    _plot_training_curves(history, curve_png)

    print(f"[INFO] Saved CoOp prompt embeddings to: {args.out_path}")
    print(f"[INFO] Saved metadata JSON to: {meta_path}")
    print(f"[INFO] Saved training history CSV: {history_csv}")
    print(f"[INFO] Saved training history JSON: {history_json}")
    print(f"[INFO] Saved training curves plot: {curve_png}")


if __name__ == "__main__":
    main()
