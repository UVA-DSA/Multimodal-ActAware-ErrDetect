"""
Models for finetuning a vision backbone on JIGSAWS labels and extracting hidden features.

Supported label types:
- gesture: 8-way gesture classification
- error: binary error / no-error classification

Supported backbones:
- ResNet (torchvision): frozen backbone + projected head + classification head
- CLIP vision encoder (transformers): LoRA finetuning of attention projection layers + classification head

Hidden features:
- ResNet: penultimate pooled embedding before classifier head, or a projected 128-d head feature
  for frozen-backbone error finetuning
- CLIP: pooled vision embedding (pooler_output) BEFORE classifier head
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn


LABEL_NUM_CLASSES = {
    "gesture": 8,
    "error": 2,
}
DEFAULT_ERROR_RESNET_PROJECTED_DIM = 128


def resolve_num_classes(label_type: str, num_classes: Optional[int] = None) -> int:
    label_key = str(label_type).lower()
    if label_key not in LABEL_NUM_CLASSES:
        raise ValueError(f"Unsupported label_type: {label_type}. Expected one of {tuple(LABEL_NUM_CLASSES.keys())}")
    if num_classes is not None:
        return int(num_classes)
    return int(LABEL_NUM_CLASSES[label_key])


@dataclass(frozen=True)
class PromptFinetuneConfig:
    backbone: str
    label_type: str = "gesture"
    num_classes: Optional[int] = None
    # Optional projected feature dim before classifier
    resnet_projected_dim: Optional[int] = None
    # Dropout regularization for the projected ResNet head
    resnet_head_dropout: float = 0.0
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    @property
    def resolved_num_classes(self) -> int:
        return resolve_num_classes(self.label_type, self.num_classes)


# Backward-compatible alias used by existing scripts.
GestureFinetuneConfig = PromptFinetuneConfig


class LoRALinear(nn.Module):
    """
    Minimal LoRA adapter for a Linear layer:
        y = W x + scale * (B (A x))
    where W is frozen, A and B are trainable low-rank matrices.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects a nn.Linear base layer")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # A: down projection (out=r, in=in_features), B: up projection (out=out_features, in=r)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)

        # Init per LoRA paper: A random, B zeros so start is no-op
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out + lora_out


def _iter_named_parameters_in_order(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    # named_parameters already yields in module registration order (good enough for our ratio-based freezing)
    for name, p in model.named_parameters():
        yield name, p


def set_trainable_ratio(model: nn.Module, trainable_ratio: float) -> None:
    """
    Freeze ~ (1-trainable_ratio) parameters by parameter-count order.
    Keeps last fraction trainable.
    """
    if not (0.0 < trainable_ratio <= 1.0):
        raise ValueError("trainable_ratio must be in (0, 1]")

    params = list(_iter_named_parameters_in_order(model))
    total = sum(p.numel() for _, p in params)
    cutoff = int(total * (1.0 - trainable_ratio))

    seen = 0
    for _, p in params:
        n = p.numel()
        # Freeze until we cross cutoff, then train remaining
        if seen + n <= cutoff:
            p.requires_grad = False
        else:
            p.requires_grad = True
        seen += n


def apply_lora_to_clip_vision(
    vision_model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_linear_names: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
) -> int:
    """
    Apply LoRA to CLIPVisionModel attention projection layers by replacing matching nn.Linear modules.
    Returns number of modules replaced.
    """
    replaced = 0

    def _replace_in_module(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in target_linear_names:
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                _replace_in_module(child)

    _replace_in_module(vision_model)
    return replaced


class ResNetPromptFinetuner(nn.Module):
    """
    ResNet backbone that returns (hidden, logits) per frame.
    Input: images (B,T,C,H,W)
    Hidden: (B,T,D)
    Logits: (B,T,num_classes)
    """

    def __init__(
        self,
        resnet_type: str = "resnet50",
        num_classes: int = 8,
        pretrained: bool = True,
        projected_dim: Optional[int] = None,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        from torchvision import models

        if resnet_type == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 512
        elif resnet_type == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            hidden_dim = 2048
        elif resnet_type == "resnet101":
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
            hidden_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Replace classifier with identity; we add our own head.
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.backbone_output_dim = hidden_dim
        self.projected_dim = int(projected_dim) if projected_dim is not None else None

        if self.projected_dim is not None:
            self.feature_proj = nn.Sequential(
                nn.Linear(hidden_dim, self.projected_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(head_dropout)) if float(head_dropout) > 0 else nn.Identity(),
            )
            self.hidden_dim = self.projected_dim
        else:
            self.feature_proj = None
            self.hidden_dim = hidden_dim

        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = images.shape
        x = images.view(b * t, c, h, w)
        hidden = self.backbone(x)  # (B*T, D_backbone)
        if self.feature_proj is not None:
            hidden = self.feature_proj(hidden)  # (B*T, D_projected)
        logits = self.classifier(hidden)  # (B*T, C)
        hidden = hidden.view(b, t, -1)
        logits = logits.view(b, t, -1)
        return hidden, logits


class ClipVisionPromptFinetuner(nn.Module):
    """
    CLIP vision backbone that returns (hidden, logits) per frame.
    Input: pixel_values (B,T,3,H,W) after CLIPProcessor; we flatten frames inside forward.
    Hidden: pooler_output (B,T,D)
    Logits: (B,T,num_classes)
    """

    def __init__(self, hf_name: str, num_classes: int = 8):
        super().__init__()
        from transformers import CLIPVisionModel

        self.vision = CLIPVisionModel.from_pretrained(hf_name)
        self.hidden_dim = self.vision.config.hidden_size
        self.classifier = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = pixel_values.shape
        x = pixel_values.view(b * t, c, h, w)
        out = self.vision(pixel_values=x)
        hidden = out.pooler_output  # (B*T, D)
        logits = self.classifier(hidden)  # (B*T, C)
        hidden = hidden.view(b, t, -1)
        logits = logits.view(b, t, -1)
        return hidden, logits


def build_prompt_finetune_model(cfg: PromptFinetuneConfig) -> Tuple[nn.Module, Dict]:
    """
    Returns (model, meta) where meta contains backbone_name and hidden_dim.
    """
    num_classes = cfg.resolved_num_classes
    label_type = str(cfg.label_type).lower()

    if cfg.backbone.startswith("resnet"):
        projected_dim = cfg.resnet_projected_dim
        model = ResNetPromptFinetuner(
            resnet_type=cfg.backbone,
            num_classes=num_classes,
            pretrained=True,
            projected_dim=projected_dim,
            head_dropout=cfg.resnet_head_dropout,
        )
        meta = {
            "backbone": cfg.backbone,
            "hidden_dim": model.hidden_dim,
            "backbone_output_dim": model.backbone_output_dim,
            "projected_dim": model.projected_dim,
            "head_dropout": float(cfg.resnet_head_dropout),
            "type": "resnet",
            "label_type": label_type,
            "num_classes": num_classes,
        }
        return model, meta

    if cfg.backbone.startswith("clip-vit"):
        # map short names to HF names used elsewhere in repo
        hf_map = {
            "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
            "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
            "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
            "clip-vit-large-patch14-336": "openai/clip-vit-large-patch14-336",
        }
        if cfg.backbone not in hf_map:
            raise ValueError(f"Unknown CLIP backbone: {cfg.backbone}")
        model = ClipVisionPromptFinetuner(hf_name=hf_map[cfg.backbone], num_classes=num_classes)
        meta = {
            "backbone": cfg.backbone,
            "hidden_dim": model.hidden_dim,
            "type": "clip",
            "hf_name": hf_map[cfg.backbone],
            "label_type": label_type,
            "num_classes": num_classes,
        }
        return model, meta

    raise ValueError(f"Unsupported backbone: {cfg.backbone}")


# Backward-compatible aliases used by existing scripts.
ResNetGestureFinetuner = ResNetPromptFinetuner
ClipVisionGestureFinetuner = ClipVisionPromptFinetuner
build_gesture_finetune_model = build_prompt_finetune_model


