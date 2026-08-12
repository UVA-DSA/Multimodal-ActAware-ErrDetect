"""
CoG's Gestural-Visual Reasoning (GVR) module adapted to window-level prediction.

The Chain-of-Gesture model (Shao et al., RA-L 2024) predicts per frame: its GVR
transformer produces one representation per frame, which the multi-scale temporal
reasoning stack then refines. This module keeps **only** the GVR transformer and
attaches a window-level head, so it emits a single error probability per input
window — the same output granularity as our Activity Prompting model. That makes
the two directly comparable under an identical data pipeline, window geometry,
label rule and evaluation protocol; the only thing that differs is the module.

Two implementation notes:

* CoG's `MyTransformer.forward` hardcodes a batch size of 1 (it builds the causal
  sliding windows with a Python loop and `squeeze(1)`). The forward below is a
  batched, vectorised equivalent that produces exactly the same windows: frame *i*
  attends to the `len_q` frames ending at *i*, zero-padded on the left when fewer
  than `len_q` frames precede it. CoG's own submodules (`linear1`, `linear2`,
  `transformer`) are reused unchanged, so the computation is CoG's.
* `len_q` defaults to the window length. CoG uses 40 over whole videos; when the
  model only ever sees one window, any `len_q` beyond the window length just adds
  zero-padded positions.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

_COG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "baselines", "Chain-of-Gesture")
)
if _COG_DIR not in sys.path:
    sys.path.insert(0, _COG_DIR)

from transformer_cot import MyTransformer  # noqa: E402


class CoGGVRWindowFeatures(nn.Module):
    """
    CoG GVR module + window-level head.

    Input:  spatial_features (B, T, D) pre-extracted frame features
    Output: logits (B, 1), one error prediction per window
    """

    def __init__(
        self,
        gesture_prompts,
        d_text: int = 512,
        d_model: int = 2048,
        d_cog: int = 64,
        segment_length: int = 10,
        len_q: Optional[int] = None,
        dropout: float = 0.1,
        prompt_features: Optional[torch.Tensor] = None,
        pool: str = "mean",
        device: str = "cpu",
    ):
        super().__init__()
        self.segment_length = int(segment_length)
        self.len_q = int(len_q) if len_q else self.segment_length
        self.d_cog = int(d_cog)
        if pool not in ("mean", "last"):
            raise ValueError(f"Unknown pool: {pool!r} (expected 'mean' or 'last')")
        self.pool = pool

        # Prompt embeddings: identical source to our model, so only the architecture differs.
        if prompt_features is not None:
            if not torch.is_tensor(prompt_features):
                raise TypeError("prompt_features must be a torch.Tensor")
            gf = prompt_features.detach().to(torch.float32)
        else:
            gf = self._clip_prompt_features(gesture_prompts)
        self.num_gestures = int(gf.size(0))
        d_text = int(gf.size(1))
        self.register_buffer("gesture_features", gf)

        # CoG's own GVR transformer (its linear1/linear2/transformer are used directly).
        self.cot = MyTransformer(d_model, d_text, self.d_cog, max(1, self.d_cog // 8), self.len_q, device)

        self.dropout = nn.Dropout(dropout)
        self.pred = nn.Linear(self.num_gestures * self.d_cog, 1)

    @staticmethod
    def _clip_prompt_features(gesture_prompts) -> torch.Tensor:
        from transformers import CLIPTextModel, CLIPTokenizer

        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        feats = []
        with torch.no_grad():
            for prompt in gesture_prompts:
                feats.append(model(**tok(prompt, return_tensors="pt")).pooler_output.squeeze(0))
        out = torch.stack(feats).detach().to(torch.float32)
        del model, tok
        return out

    def forward(self, spatial_features: torch.Tensor, masks: Optional[torch.Tensor] = None):
        b, t, _ = spatial_features.shape

        # Pad / truncate the window to segment_length, mirroring our model's handling.
        if t < self.segment_length:
            pad = torch.zeros(
                b, self.segment_length - t, spatial_features.size(-1),
                device=spatial_features.device, dtype=spatial_features.dtype,
            )
            spatial_features = torch.cat([spatial_features, pad], dim=1)
            if masks is not None:
                mpad = torch.zeros(b, self.segment_length - t, device=masks.device, dtype=torch.bool)
                masks = torch.cat([masks.to(torch.bool), mpad], dim=1)
        elif t > self.segment_length:
            spatial_features = spatial_features[:, : self.segment_length, :]
            if masks is not None:
                masks = masks[:, : self.segment_length]
        t = self.segment_length

        visual = self.cot.linear1(spatial_features)          # (B, T, d_cog)
        text = self.cot.linear2(self.gesture_features)       # (J, d_cog)

        # Causal sliding windows: frame i sees the len_q frames ending at i,
        # left zero-padded. Vectorised equivalent of CoG's per-frame loop.
        pad = torch.zeros(b, self.len_q - 1, self.d_cog, device=visual.device, dtype=visual.dtype)
        padded = torch.cat([pad, visual], dim=1)                       # (B, T+len_q-1, d)
        wins = padded.unfold(1, self.len_q, 1)                         # (B, T, d, len_q)
        wins = wins.permute(0, 1, 3, 2).reshape(b * t, self.len_q, self.d_cog)

        txt = text.unsqueeze(0).expand(b * t, -1, -1)                  # (B*T, J, d_cog)
        out = self.cot.transformer(wins, txt)                          # (B*T, J, d_cog)
        out = out.reshape(b, t, self.num_gestures * self.d_cog)        # (B, T, J*d_cog)

        # Window-level pooling over time -> a single prediction per window.
        if self.pool == "last":
            pooled = out[:, -1, :]
        elif masks is not None:
            valid = masks.to(out.dtype).unsqueeze(-1)
            pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = out.mean(dim=1)

        return self.pred(self.dropout(pooled))
