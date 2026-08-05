"""
GVR Module with Pre-extracted Features

These model variants take pre-extracted ResNet features as input instead of raw images,
avoiding redundant CNN computation during training and evaluation.

Use these models with dataloader_features.py after running extract_features.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import CLIPTextModel, CLIPTokenizer


KIN_DIV = 8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure positional encoding matches x dimensions (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class AdditiveAttnPool(nn.Module):
    """Learned additive attention pooling over the time dimension."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D); key_padding_mask: (B, T) True = masked
        scores = self.score(x).squeeze(-1)  # (B, T)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


def _masked_mean_pool(x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # x: (B, T, D); key_padding_mask: (B, T) True = masked
    if key_padding_mask is None:
        return x.mean(dim=1)
    valid = (~key_padding_mask).to(x.dtype).unsqueeze(-1)  # (B, T, 1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (x * valid).sum(dim=1) / denom


def _build_pooled_branch(pooled_mode: str, d_text: int):
    """Return the module(s) used by the direct sequence-summary branch."""
    if pooled_mode == "tcn":
        return nn.Sequential(
            nn.Conv1d(d_text, d_text, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
    if pooled_mode == "attn":
        return AdditiveAttnPool(d_text)
    if pooled_mode == "mean":
        return None
    raise ValueError(f"Unknown pooled_mode: {pooled_mode!r} (expected 'tcn', 'mean' or 'attn')")


class GVRModulePredFeatures(nn.Module):
    """
    GVR Module that takes pre-extracted CNN features as input.
    
    Input: spatial_features of shape (batch_size, seq_len, feature_dim)
           where feature_dim is typically 2048 for ResNet50
    """
    def __init__(
        self,
        gesture_prompts,
        d_text,
        d_model=2048,
        num_heads=1,
        segment_length=20,
        position=False,
        dropout=0.1,
        prompt_features: Optional[torch.Tensor] = None,
        use_pooled_branch: bool = True,
        fusion: str = "concat",
        pooled_mode: str = "tcn",
        attn_temperature: float = 1.0,
    ):
        super(GVRModulePredFeatures, self).__init__()
        self.use_positional_encoding = position
        self.num_gestures = len(gesture_prompts)
        self.segment_length = segment_length
        self.d_text = d_text
        self.use_pooled_branch = use_pooled_branch
        if fusion not in ("concat", "gated", "film"):
            raise ValueError(f"Unknown fusion: {fusion!r} (expected 'concat', 'gated' or 'film')")
        self.fusion = fusion
        self.pooled_mode = pooled_mode
        self.attn_temperature = float(attn_temperature)
        self.act = nn.GELU()
        
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_text, max_len=segment_length)
        
        # Prompt projection for text-encoded gesture features
        self.prompt_proj = nn.Linear(d_text, d_text)
        self.prompt_act = nn.GELU()
        self.gesture_context_proj = nn.Linear(2 * d_text, d_text)
        self.gesture_context_act = nn.GELU()

        # Generate gesture features using CLIP (done once during init) or load precomputed features
        if prompt_features is not None:
            if not torch.is_tensor(prompt_features):
                raise TypeError("prompt_features must be a torch.Tensor")
            if prompt_features.size(0) != self.num_gestures:
                raise ValueError(
                    f"prompt_features length mismatch: expected {self.num_gestures}, got {prompt_features.size(0)}"
                )
            if prompt_features.size(1) != d_text:
                raise ValueError(f"prompt_features dim mismatch: expected {d_text}, got {prompt_features.size(1)}")
            self.register_buffer("gesture_features", prompt_features.detach())
        else:
            self._init_gesture_features(gesture_prompts, d_text)
        
        # Linear layers and normalization (no .to(device) - will move with model)
        self.projection = nn.Linear(d_model, d_text)
        self.feed_forward = nn.Linear(d_text, d_text)
        self.layer_norm1 = nn.LayerNorm(d_text)
        self.layer_norm2 = nn.LayerNorm(d_text)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # First Attention layer for spatially-aware gesture features
        self.trans = nn.MultiheadAttention(embed_dim=d_text, num_heads=num_heads, 
                                           dropout=dropout, batch_first=True)
        
        # Second Attention layer for refined gesture features
        self.attention = nn.MultiheadAttention(embed_dim=d_text, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        
        self.bottleneck = nn.Linear(self.num_gestures * d_text, 256)

        # Direct video summary path (forces dependence on visual features)
        self.seq_reducer = _build_pooled_branch(self.pooled_mode, d_text)
        self.video_fc = nn.Sequential(
            nn.LayerNorm(d_text),
            nn.Linear(d_text, d_text),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion of refined prompt features and pooled video summary
        if self.fusion == "gated":
            self.fusion_gate = nn.Linear(2 * d_text, d_text)
        elif self.fusion == "film":
            self.film_proj = nn.Linear(d_text, 2 * d_text)

        # Define prediction layer based on gestures + pooled video summary
        if self.fusion == "gated":
            pred_in_dim = self.num_gestures * d_text + d_text
        else:
            pred_in_dim = self.num_gestures * d_text + (d_text if self.use_pooled_branch else 0)
        self.pred = nn.Linear(pred_in_dim, 1)

    def _init_gesture_features(self, gesture_prompts, d_text):
        """
        Generate gesture prompt features {gj} using CLIP's text encoder.
        Registers them as a buffer so they move with the model.
        """
        # Use CPU for initial generation, will move with model.to(device)
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

        gesture_features = []
        with torch.no_grad():
            for gesture in gesture_prompts:
                inputs = tokenizer(gesture, return_tensors="pt")
                gesture_feature = clip_model(**inputs).pooler_output.squeeze(0)
                gesture_features.append(gesture_feature)

        # Register as buffer so it moves with model.to(device)
        gesture_tensor = torch.stack(gesture_features).detach()
        self.register_buffer('gesture_features', gesture_tensor)

        # Clean up CLIP model (not needed after feature extraction)
        del clip_model
        del tokenizer

    def _pool_sequence(self, seq_features, key_padding_mask):
        """Compute the pooled sequence summary according to `pooled_mode`."""
        if self.pooled_mode == "tcn":
            if key_padding_mask is not None:
                valid_mask = (~key_padding_mask).to(seq_features.dtype).unsqueeze(-1)
                seq_features = seq_features * valid_mask
            return self.seq_reducer(seq_features.transpose(1, 2)).squeeze(-1)
        if self.pooled_mode == "attn":
            return self.seq_reducer(seq_features, key_padding_mask)
        return _masked_mean_pool(seq_features, key_padding_mask)

    def _fuse_and_predict(self, refined_ct, pooled):
        """Combine refined prompt features with the pooled summary and predict."""
        batch_size = refined_ct.size(0)
        if self.fusion == "film":
            gamma, beta = self.film_proj(pooled).chunk(2, dim=-1)
            refined_ct = refined_ct * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        ct = refined_ct.reshape(batch_size, -1)
        ct = self.dropout(ct)
        if self.fusion == "gated":
            mean_ct = refined_ct.mean(dim=1)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([mean_ct, pooled], dim=-1)))
            fused = gate * pooled + (1.0 - gate) * mean_ct
            return self.pred(torch.cat([ct, fused], dim=1))
        if self.use_pooled_branch:
            return self.pred(torch.cat([ct, pooled], dim=1))
        return self.pred(ct)

    def forward(self, spatial_features, masks=None, return_attn: bool = False):
        """
        Forward pass using pre-extracted spatial features.
        
        Args:
            spatial_features: Tensor of shape (batch_size, seq_len, feature_dim)
                             Pre-extracted ResNet features for each frame
            masks: Optional tensor of shape (batch_size, seq_len) indicating valid positions (1=valid, 0=padding)
        """
        batch_size, seq_len, feature_dim = spatial_features.shape
        
        # The features are already extracted, so we just need to handle padding/truncation
        spatial_embeddings = spatial_features
        
        # key_padding_mask: (batch_size, seq_len) where True = masked (ignored)
        # masks expected as (batch_size, seq_len) with True=valid, False=padding
        if masks is not None:
            masks = masks.to(torch.bool)
            key_padding_mask = ~masks
        else:
            key_padding_mask = None
        
        # Pad or truncate to segment_length if necessary
        if seq_len < self.segment_length:
            # Pad with zeros if sequence is shorter
            padding = torch.zeros(batch_size, self.segment_length - seq_len, feature_dim,
                                  device=spatial_embeddings.device, dtype=spatial_embeddings.dtype)
            spatial_embeddings = torch.cat([spatial_embeddings, padding], dim=1)
            # Also pad the key_padding_mask
            if key_padding_mask is not None:
                mask_padding = torch.ones(batch_size, self.segment_length - seq_len,
                                          device=key_padding_mask.device, dtype=torch.bool)
                key_padding_mask = torch.cat([key_padding_mask, mask_padding], dim=1)
            else:
                # Create mask where original positions are valid, padded positions are masked
                key_padding_mask = torch.zeros(batch_size, self.segment_length, device=spatial_embeddings.device, dtype=torch.bool)
                key_padding_mask[:, seq_len:] = True  # Mask the padded positions
        elif seq_len > self.segment_length:
            # Truncate if sequence is longer
            spatial_embeddings = spatial_embeddings[:, :self.segment_length, :]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, :self.segment_length]
        
        # Project spatial embeddings to d_text dimension
        spatial_embeddings = self.projection(spatial_embeddings)  # Shape: (batch_size, segment_length, d_text)
        spatial_embeddings = self.act(spatial_embeddings)
        spatial_embeddings = self.dropout(spatial_embeddings)  # Apply dropout after projection
        
        if self.use_positional_encoding:
            spatial_embeddings = self.positional_encoding(spatial_embeddings)
        
        gesture_features = self.prompt_act(self.prompt_proj(self.gesture_features))
        gesture_query = gesture_features.unsqueeze(0).expand(batch_size, -1, -1)  # (B, J, d_text)
        # Dividing the query by the temperature scales the attention logits by
        # 1/temperature, so values < 1 sharpen the prompt-to-frame attention.
        attn_output, attn_weights = self.trans(
            query=gesture_query / self.attn_temperature,
            key=spatial_embeddings,
            value=spatial_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attn_output = self.layer_norm1(attn_output + gesture_query)
        ff_output = self.feed_forward(attn_output)
        ff_output = self.act(ff_output)
        ff_output = self.dropout(ff_output)
        spatial_aware_gestures = self.layer_norm2(ff_output + attn_output)

        pooled = self._pool_sequence(spatial_embeddings, key_padding_mask)  # (B, d_text)
        pooled = self.video_fc(pooled)

        # Apply secondary attention with pooled-context gesture features as key/value
        gesture_expanded = gesture_features.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, self.num_gestures, -1)
        gesture_expanded = self.gesture_context_act(
            self.gesture_context_proj(torch.cat([gesture_expanded, pooled_expanded], dim=-1))
        )
        refined_ct, _ = self.attention(
            query=spatial_aware_gestures,
            key=gesture_expanded,
            value=gesture_expanded
        )

        # Fuse refined prompt features with the pooled summary and predict
        out = self._fuse_and_predict(refined_ct, pooled)

        if return_attn:
            return out, attn_weights
        return out


class GVRModuleContexPredFeatures(nn.Module):
    """
    GVR Context Module that takes pre-extracted CNN features as input.
    
    Input: spatial_features of shape (batch_size, seq_len, feature_dim)
           where feature_dim is typically 2048 for ResNet50
    """
    def __init__(self, d_model_ges, d_model=2048, num_heads=1, segment_length=40,
                 layer_norm1=True, layer_norm2=True, layer_norm3=True, position=False, dropout=0.1):
        super(GVRModuleContexPredFeatures, self).__init__()
        self.use_positional_encoding = position
        self.segment_length = segment_length
        self.act = nn.GELU()
        
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model_ges, max_len=segment_length)
        
        # No CNN needed - features are pre-extracted
        self.cnn_out_dim = d_model
        
        # Linear layer to project CNN output to d_model_ges dimension
        self.projectioncnn = nn.Linear(d_model, d_model_ges)
        self.projectioncnnges = nn.Linear(d_model, d_model_ges)
        self.layer_norm1_exist = layer_norm1
        self.layer_norm2_exist = layer_norm2
        self.layer_norm3_exist = layer_norm3
        self.layer_norm1 = nn.LayerNorm(d_model_ges)
        self.layer_norm2 = nn.LayerNorm(d_model_ges)
        self.layer_norm3 = nn.LayerNorm(d_model_ges)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        self.trans = nn.MultiheadAttention(embed_dim=d_model_ges, num_heads=num_heads, 
                                           dropout=dropout, batch_first=True)
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=d_model_ges, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        # Prediction layer
        self.pred = nn.Linear(d_model_ges, 1)

    def forward(self, spatial_features):
        """
        Forward pass using pre-extracted spatial features.
        
        Args:
            spatial_features: Tensor of shape (batch_size, seq_len, feature_dim)
                             Pre-extracted ResNet features for each frame
        """
        batch_size, seq_len, feature_dim = spatial_features.shape
        
        # Handle case where we have more frames than segment_length
        if seq_len > self.segment_length:
            spatial_features = spatial_features[:, :self.segment_length, :]
            seq_len = self.segment_length
        elif seq_len < self.segment_length:
            # Pad if we have fewer frames
            padding = torch.zeros(batch_size, self.segment_length - seq_len, feature_dim,
                                  device=spatial_features.device, dtype=spatial_features.dtype)
            spatial_features = torch.cat([spatial_features, padding], dim=1)
            seq_len = self.segment_length
        
        # Project features for both branches
        spatial_embeddings = self.projectioncnn(spatial_features)  # Shape: (batch_size, segment_length, d_model_ges)
        spatial_embeddings = self.act(spatial_embeddings)
        spatial_embeddings = self.dropout(spatial_embeddings)
        spatial_embeddings_ges = self.projectioncnnges(spatial_features)  # Shape: (batch_size, segment_length, d_model_ges)
        spatial_embeddings_ges = self.act(spatial_embeddings_ges)
        spatial_embeddings_ges = self.dropout(spatial_embeddings_ges)
        
        if self.use_positional_encoding:
            spatial_embeddings = self.positional_encoding(spatial_embeddings)
            spatial_embeddings_ges = self.positional_encoding(spatial_embeddings_ges)
        
        attout, _ = self.trans(spatial_embeddings_ges, spatial_embeddings, spatial_embeddings)
        if self.layer_norm1_exist:
            attout = self.layer_norm1(attout + spatial_embeddings_ges)
        
        # Apply attention mechanism
        spatial_aware_gesture, _ = self.attention(attout, spatial_embeddings_ges, spatial_embeddings_ges)
        if self.layer_norm2_exist:
            spatial_aware_gesture = self.layer_norm2(spatial_aware_gesture + attout)
        if self.layer_norm3_exist:
            spatial_aware_gesture = self.layer_norm3(spatial_aware_gesture + spatial_embeddings)
        
        spatial_aware_gesture = self.dropout(spatial_aware_gesture)
        
        # Apply prediction to each frame separately
        out = self.pred(spatial_aware_gesture)  # Shape: (batch_size, segment_length, 1)
        
        return out.squeeze(-1)  # Shape: (batch_size, segment_length)


class GVRModuleContexPredDualFeatures(nn.Module):
    """
    Context model variant that consumes TWO pre-extracted feature streams:
      - base_features: output of extract_features.py (resnet/clip frozen encoder)
      - ges_features: output of extract_gesture_prompt_features.py (finetuned gesture model hidden)

    This replaces `GVRModuleContexPred`'s cnn/cnnges modules.
    """

    def __init__(
        self,
        base_dim: int,
        ges_dim: int,
        d_embed: int,
        num_heads: int = 1,
        segment_length: int = 20,
        layer_norm1: bool = True,
        layer_norm2: bool = True,
        layer_norm3: bool = True,
        position: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.segment_length = segment_length
        self.use_positional_encoding = position
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_embed, max_len=segment_length)

        self.proj_base = nn.Linear(base_dim, d_embed)
        self.proj_ges = nn.Linear(ges_dim, d_embed)
        self.act = nn.GELU()

        self.layer_norm1_exist = layer_norm1
        self.layer_norm2_exist = layer_norm2
        self.layer_norm3_exist = layer_norm3
        self.layer_norm1 = nn.LayerNorm(d_embed)
        self.layer_norm2 = nn.LayerNorm(d_embed)
        self.layer_norm3 = nn.LayerNorm(d_embed)

        self.dropout = nn.Dropout(dropout)

        self.trans = nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.pred = nn.Linear(d_embed, 1)

    def forward(self, base_features: torch.Tensor, ges_features: torch.Tensor, masks: torch.Tensor | None = None):
        """
        Args:
            base_features: (B,T,Db)
            ges_features:  (B,T,Dg)
            masks: optional (B,T) bool where True=valid
        Returns:
            logits: (B,segment_length)
        """
        b, t, _ = base_features.shape
        # Truncate/pad both streams to same length
        t_min = min(base_features.size(1), ges_features.size(1))
        base_features = base_features[:, :t_min, :]
        ges_features = ges_features[:, :t_min, :]
        if masks is not None:
            masks = masks[:, :t_min]

        if t_min < self.segment_length:
            pad_len = self.segment_length - t_min
            base_pad = torch.zeros(b, pad_len, base_features.size(-1), device=base_features.device, dtype=base_features.dtype)
            ges_pad = torch.zeros(b, pad_len, ges_features.size(-1), device=ges_features.device, dtype=ges_features.dtype)
            base_features = torch.cat([base_features, base_pad], dim=1)
            ges_features = torch.cat([ges_features, ges_pad], dim=1)
            if masks is not None:
                mask_pad = torch.zeros(b, pad_len, device=masks.device, dtype=torch.bool)
                masks = torch.cat([masks.to(torch.bool), mask_pad], dim=1)
        elif t_min > self.segment_length:
            base_features = base_features[:, : self.segment_length, :]
            ges_features = ges_features[:, : self.segment_length, :]
            if masks is not None:
                masks = masks[:, : self.segment_length]

        # Project
        base_emb = self.act(self.proj_base(base_features))
        base_emb = self.dropout(base_emb)
        ges_emb = self.act(self.proj_ges(ges_features))
        ges_emb = self.dropout(ges_emb)

        if self.use_positional_encoding:
            base_emb = self.positional_encoding(base_emb)
            ges_emb = self.positional_encoding(ges_emb)

        key_padding_mask = None
        if masks is not None:
            key_padding_mask = ~masks.to(torch.bool)  # True = masked

        # Cross-attn: query=gesture-branch, key/value=base-branch
        attout, _ = self.trans(query=ges_emb, key=base_emb, value=base_emb, key_padding_mask=key_padding_mask)
        if self.layer_norm1_exist:
            attout = self.layer_norm1(attout + ges_emb)

        spatial_aware, _ = self.attention(query=attout, key=ges_emb, value=ges_emb, key_padding_mask=key_padding_mask)
        if self.layer_norm2_exist:
            spatial_aware = self.layer_norm2(spatial_aware + attout)
        if self.layer_norm3_exist:
            spatial_aware = self.layer_norm3(spatial_aware + base_emb)

        spatial_aware = self.dropout(spatial_aware)
        out = self.pred(spatial_aware).squeeze(-1)  # (B,T)
        return out


class GVRModulePredKinFeatures(nn.Module):
    """
    GVR Module with kinematics that takes pre-extracted CNN features as input.
    
    Input: spatial_features of shape (batch_size, seq_len, feature_dim)
           kine of shape (batch_size, segment_length, 14) for kinematics data
    """
    def __init__(
        self,
        gesture_prompts,
        d_text,
        d_model=2048,
        num_heads=1,
        segment_length=20,
        position=False,
        dropout=0.1,
        prompt_features: Optional[torch.Tensor] = None,
        use_pooled_branch: bool = False,
        fusion: str = "concat",
        pooled_mode: str = "tcn",
        attn_temperature: float = 1.0,
        kin_fusion: str = "concat",
        kin_dim_ratio: int = KIN_DIV,
    ):
        super(GVRModulePredKinFeatures, self).__init__()
        self.use_positional_encoding = position
        self.num_gestures = len(gesture_prompts)
        self.segment_length = segment_length
        self.d_text = d_text
        self.use_pooled_branch = use_pooled_branch
        if fusion not in ("concat", "gated", "film"):
            raise ValueError(f"Unknown fusion: {fusion!r} (expected 'concat', 'gated' or 'film')")
        if kin_fusion not in ("concat", "gated"):
            raise ValueError(f"Unknown kin_fusion: {kin_fusion!r} (expected 'concat' or 'gated')")
        self.fusion = fusion
        self.pooled_mode = pooled_mode
        self.attn_temperature = float(attn_temperature)
        self.kin_fusion = kin_fusion
        self.kin_dim_ratio = max(1, int(kin_dim_ratio))
        self.act = nn.GELU()
        
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_text, max_len=segment_length)
        
        # Prompt projection for text-encoded gesture features
        self.prompt_proj = nn.Linear(d_text, d_text)
        self.prompt_act = nn.GELU()
        self.gesture_context_proj = nn.Linear(2 * d_text, d_text)
        self.gesture_context_act = nn.GELU()

        # Generate gesture features using CLIP (done once during init) or load precomputed features
        if prompt_features is not None:
            if not torch.is_tensor(prompt_features):
                raise TypeError("prompt_features must be a torch.Tensor")
            if prompt_features.size(0) != self.num_gestures:
                raise ValueError(
                    f"prompt_features length mismatch: expected {self.num_gestures}, got {prompt_features.size(0)}"
                )
            if prompt_features.size(1) != d_text:
                raise ValueError(f"prompt_features dim mismatch: expected {d_text}, got {prompt_features.size(1)}")
            self.register_buffer("gesture_features", prompt_features.detach())
        else:
            self._init_gesture_features(gesture_prompts, d_text)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Linear layers and normalization (separate projections for spatial and kinematics)
        kin_proj_dim = max(1, d_text // self.kin_dim_ratio)
        self.spatial_proj = nn.Linear(d_model, d_text)
        if self.kin_fusion == "gated":
            # Project kinematics up to d_text and inject via a learned per-frame gate.
            self.kine_proj = nn.Linear(14, d_text)
            self.kin_gate = nn.Linear(2 * d_text, d_text)
        else:
            self.kine_proj = nn.Linear(14, kin_proj_dim)
            self.combined_proj = nn.Linear(d_text + kin_proj_dim, d_text)
        self.feed_forward = nn.Linear(d_text, d_text)
        self.layer_norm1 = nn.LayerNorm(d_text)
        self.layer_norm2 = nn.LayerNorm(d_text)

        # First Attention layer for spatially-aware gesture features
        self.trans = nn.MultiheadAttention(embed_dim=d_text, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)

        # Second Attention layer for refined gesture features
        self.attention = nn.MultiheadAttention(embed_dim=d_text, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)

        self.bottleneck = nn.Linear(self.num_gestures * d_text, 256)
        # Direct summary path (matches `GVRModulePredFeatures`'s design)
        self.seq_reducer = _build_pooled_branch(self.pooled_mode, d_text)
        self.video_fc = nn.Sequential(
            nn.LayerNorm(d_text),
            nn.Linear(d_text, d_text),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion of refined prompt features and pooled (visual+kin) summary
        if self.fusion == "gated":
            self.fusion_gate = nn.Linear(2 * d_text, d_text)
        elif self.fusion == "film":
            self.film_proj = nn.Linear(d_text, 2 * d_text)

        # Final prediction uses both gesture-conditioned features and pooled (visual+kin) summary
        if self.fusion == "gated":
            pred_in_dim = self.num_gestures * d_text + d_text
        else:
            pred_in_dim = self.num_gestures * d_text + (d_text if self.use_pooled_branch else 0)
        self.pred = nn.Linear(pred_in_dim, 1)

    def _pool_sequence(self, seq_features, key_padding_mask):
        """Compute the pooled sequence summary according to `pooled_mode`."""
        if self.pooled_mode == "tcn":
            if key_padding_mask is not None:
                valid_mask = (~key_padding_mask).to(seq_features.dtype).unsqueeze(-1)
                seq_features = seq_features * valid_mask
            return self.seq_reducer(seq_features.transpose(1, 2)).squeeze(-1)
        if self.pooled_mode == "attn":
            return self.seq_reducer(seq_features, key_padding_mask)
        return _masked_mean_pool(seq_features, key_padding_mask)

    def _fuse_and_predict(self, refined_ct, pooled):
        """Combine refined prompt features with the pooled summary and predict."""
        batch_size = refined_ct.size(0)
        if self.fusion == "film":
            gamma, beta = self.film_proj(pooled).chunk(2, dim=-1)
            refined_ct = refined_ct * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        ct = refined_ct.reshape(batch_size, -1)
        ct = self.dropout(ct)
        if self.fusion == "gated":
            mean_ct = refined_ct.mean(dim=1)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([mean_ct, pooled], dim=-1)))
            fused = gate * pooled + (1.0 - gate) * mean_ct
            return self.pred(torch.cat([ct, fused], dim=1))
        if self.use_pooled_branch:
            return self.pred(torch.cat([ct, pooled], dim=1))
        return self.pred(ct)

    def _init_gesture_features(self, gesture_prompts, d_text):
        """
        Generate gesture prompt features {gj} using CLIP's text encoder.
        Registers them as a buffer so they move with the model.
        """
        # Use CPU for initial generation, will move with model.to(device)
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        
        gesture_features = []
        with torch.no_grad():
            for gesture in gesture_prompts:
                inputs = tokenizer(gesture, return_tensors="pt")
                gesture_feature = clip_model(**inputs).pooler_output.squeeze(0)
                gesture_features.append(gesture_feature)
        
        # Register as buffer so it moves with model.to(device)
        gesture_tensor = torch.stack(gesture_features).detach()
        self.register_buffer('gesture_features', gesture_tensor)
        
        # Clean up CLIP model (not needed after feature extraction)
        del clip_model
        del tokenizer

    def forward(self, spatial_features, kine, masks=None):
        """
        Forward pass using pre-extracted spatial features and kinematics.
        
        Args:
            spatial_features: Tensor of shape (batch_size, seq_len, feature_dim)
                             Pre-extracted ResNet features for each frame
            kine: Tensor of shape (batch_size, seq_len, 14)
                  Kinematics data
            masks: Optional bool tensor of shape (batch_size, seq_len) where True=valid, False=padding
        """
        batch_size, seq_len, feature_dim = spatial_features.shape

        # Align lengths across streams (defensive)
        t_min = min(spatial_features.size(1), kine.size(1))
        spatial_features = spatial_features[:, :t_min, :]
        kine = kine[:, :t_min, :]
        if masks is not None:
            masks = masks[:, :t_min].to(torch.bool)

        # key_padding_mask: True = masked (ignored)
        key_padding_mask = None
        if masks is not None:
            key_padding_mask = ~masks

        # Pad/truncate to segment_length
        t = spatial_features.size(1)
        if t < self.segment_length:
            pad_len = self.segment_length - t
            feat_pad = torch.zeros(batch_size, pad_len, feature_dim, device=spatial_features.device, dtype=spatial_features.dtype)
            kine_pad = torch.zeros(batch_size, pad_len, kine.size(-1), device=kine.device, dtype=kine.dtype)
            spatial_features = torch.cat([spatial_features, feat_pad], dim=1)
            kine = torch.cat([kine, kine_pad], dim=1)
            if key_padding_mask is not None:
                mask_pad = torch.ones(batch_size, pad_len, device=key_padding_mask.device, dtype=torch.bool)
                key_padding_mask = torch.cat([key_padding_mask, mask_pad], dim=1)
            else:
                key_padding_mask = torch.zeros(batch_size, self.segment_length, device=spatial_features.device, dtype=torch.bool)
                key_padding_mask[:, t:] = True
        elif t > self.segment_length:
            spatial_features = spatial_features[:, : self.segment_length, :]
            kine = kine[:, : self.segment_length, :]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, : self.segment_length]

        spatial_proj = self.act(self.spatial_proj(spatial_features))
        if self.kin_fusion == "gated":
            kin_up = self.act(self.kine_proj(kine))  # (B, T, d_text)
            gate = torch.sigmoid(self.kin_gate(torch.cat([spatial_proj, kin_up], dim=-1)))
            spatial_embeddings = spatial_proj + gate * kin_up
        else:
            kine_proj = self.kine_proj(kine)
            combined = torch.cat((spatial_proj, kine_proj), dim=-1)
            # Project fused embeddings to d_text dimension
            spatial_embeddings = self.combined_proj(combined)
            spatial_embeddings = self.act(spatial_embeddings)
        spatial_embeddings = self.dropout(spatial_embeddings)

        if self.use_positional_encoding:
            spatial_embeddings = self.positional_encoding(spatial_embeddings)

        gesture_features = self.prompt_act(self.prompt_proj(self.gesture_features))
        gesture_query = gesture_features.unsqueeze(0).expand(batch_size, -1, -1)
        # Dividing the query by the temperature scales the attention logits by
        # 1/temperature, so values < 1 sharpen the prompt-to-frame attention.
        attn_output, _ = self.trans(
            query=gesture_query / self.attn_temperature,
            key=spatial_embeddings,
            value=spatial_embeddings,
            key_padding_mask=key_padding_mask,
        )
        attn_output = self.layer_norm1(attn_output + gesture_query)
        ff_output = self.feed_forward(attn_output)
        ff_output = self.act(ff_output)
        ff_output = self.dropout(ff_output)
        spatial_aware_gestures = self.layer_norm2(ff_output + attn_output)

        pooled = self._pool_sequence(spatial_embeddings, key_padding_mask)  # (B, d_text)
        pooled = self.video_fc(pooled)

        gesture_expanded = gesture_features.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, self.num_gestures, -1)
        gesture_expanded = self.gesture_context_act(
            self.gesture_context_proj(torch.cat([gesture_expanded, pooled_expanded], dim=-1))
        )
        refined_ct, _ = self.attention(
            query=spatial_aware_gestures,
            key=gesture_expanded,
            value=gesture_expanded
        )

        out = self._fuse_and_predict(refined_ct, pooled)

        return out
