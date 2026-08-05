"""
SAR_RARP models.

This module has been updated to match the root feature-based implementations:
- Models consume pre-extracted per-frame features (not raw images).
- `GVRModulePred` outputs ONE logit per segment in a batch (segment-level prediction).
- Default model complexity matches root (`num_heads=4`, dropout, pooled video summary path).
- Attention is mask-aware (padding-safe) and sequences are padded/truncated to `n_frames`.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import models


class ResNetEmbedding(nn.Module):
    def __init__(self, resnet_type="resnet50", pretrained=True):
        super(ResNetEmbedding, self).__init__()
        
        # Load the specified ResNet model (can be resnet18, resnet50, etc.)
        if resnet_type == "resnet18":
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        # Remove the final fully connected layer to extract features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    
    def forward(self, x):

        # Extract ResNet embeddings
        embeddings = self.resnet(x)  # Shape: [(batch * seq_length), 2048, 1, 1]

        # Flatten the embeddings
        embeddings = embeddings.view(-1)  # NOTE: legacy helper; not used by feature-based training.

        return embeddings

class ResNetBinaryClassification(nn.Module):
    def __init__(self, resnet_type="resnet50", pretrained=True):
        super(ResNetBinaryClassification, self).__init__()
        
        # Load the specified ResNet model (can be resnet18, resnet50, etc.)
        if resnet_type == "resnet18":
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnet_type == "resnet50":
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        # Modify the fully connected layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)
    
    def forward(self, x):
        # x shape: [batch, seq_length, channel, width, height]
        batch_size, seq_length, channel, width, height = x.shape

        # Reshape to merge batch and seq_length dimensions
        x = x.view(batch_size * seq_length, channel, width, height)  # Shape: [(batch * seq_length), channel, width, height]

        # Pass through ResNet
        features = self.resnet(x)  # Shape: [(batch * seq_length)]


        output = features.view(batch_size, seq_length)  # Shape: [batch, seq_length]

        return output


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


    
    
class GVRModulePred(nn.Module):
    """
    Segment-level error prediction model.

    Matches root `GVRModulePredFeatures`:
    - Input: spatial_features (B, T, D) where D is the feature dim (e.g., 2048 ResNet, 768 CLIP ViT-B/32)
    - Output: logits (B, 1)  (one prediction per segment)
    - Mask-aware attention via key_padding_mask
    """

    def __init__(
        self,
        gesture_prompts,
        d_text: int = 512,
        d_model: int = 2048,
        num_heads: int = 4,
        n_frames: int = 40,
        position: bool = False,
        dropout: float = 0.1,
        precomputed_gesture_features: Optional[torch.Tensor] = None,
        disable_pooled: bool = False,
        fusion: str = "concat",
        pooled_mode: str = "mean",
        attn_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_positional_encoding = position
        self.n_frames = int(n_frames)
        self.d_text = int(d_text)
        self.act = nn.GELU()
        self.disable_pooled = bool(disable_pooled)
        if fusion not in ("concat", "gated", "film"):
            raise ValueError(f"Unknown fusion: {fusion!r} (expected 'concat', 'gated' or 'film')")
        if pooled_mode not in ("mean", "tcn", "attn"):
            raise ValueError(f"Unknown pooled_mode: {pooled_mode!r} (expected 'mean', 'tcn' or 'attn')")
        self.fusion = fusion
        self.pooled_mode = pooled_mode
        self.attn_temperature = float(attn_temperature)

        # Keep compatibility with SurgVLP-precomputed prompt embeddings.
        if precomputed_gesture_features is not None:
            gf = precomputed_gesture_features.detach()
            if gf.dim() != 2:
                raise ValueError(f"precomputed_gesture_features must be 2D (J,D), got shape={tuple(gf.shape)}")
            self.num_gestures = int(gf.size(0))
            self.d_text = int(gf.size(1))
            self.register_buffer("gesture_features", gf.to(torch.float32))
        else:
            self.num_gestures = len(gesture_prompts)
            self._init_gesture_features(gesture_prompts)

        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(self.d_text, max_len=self.n_frames)

        self.prompt_proj = nn.Linear(self.d_text, self.d_text)
        self.prompt_act = nn.GELU()
        self.projection = nn.Linear(d_model, self.d_text)
        self.feed_forward = nn.Linear(self.d_text, self.d_text)
        self.layer_norm1 = nn.LayerNorm(self.d_text)
        self.layer_norm2 = nn.LayerNorm(self.d_text)

        self.dropout = nn.Dropout(dropout)

        self.trans = nn.MultiheadAttention(embed_dim=self.d_text, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=self.d_text, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.video_fc = nn.Sequential(
            nn.LayerNorm(self.d_text),
            nn.Linear(self.d_text, self.d_text),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.pooled_mode == "tcn":
            self.seq_reducer = nn.Sequential(
                nn.Conv1d(self.d_text, self.d_text, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        elif self.pooled_mode == "attn":
            self.pool_score = nn.Sequential(
                nn.Linear(self.d_text, self.d_text), nn.Tanh(), nn.Linear(self.d_text, 1)
            )

        # Fusion of refined prompt features and pooled video summary
        if self.fusion == "gated":
            self.fusion_gate = nn.Linear(2 * self.d_text, self.d_text)
        elif self.fusion == "film":
            self.film_proj = nn.Linear(self.d_text, 2 * self.d_text)

        if self.fusion == "gated":
            self.pred = nn.Linear(self.num_gestures * self.d_text + self.d_text, 1)
        elif self.disable_pooled:
            self.pred = nn.Linear(self.num_gestures * self.d_text, 1)
        else:
            self.pred = nn.Linear(self.num_gestures * self.d_text + self.d_text, 1)

    def _pool_sequence(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooled_mode == "tcn":
            if key_padding_mask is not None:
                x = x * (~key_padding_mask).to(x.dtype).unsqueeze(-1)
            return self.seq_reducer(x.transpose(1, 2)).squeeze(-1)
        if self.pooled_mode == "attn":
            scores = self.pool_score(x).squeeze(-1)
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask, float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (x * weights).sum(dim=1)
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).to(x.dtype)
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (x * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        return x.mean(dim=1)

    def _init_gesture_features(self, gesture_prompts):
        clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        feats = []
        with torch.no_grad():
            for g in gesture_prompts:
                inputs = tokenizer(g, return_tensors="pt")
                feats.append(clip_model(**inputs).pooler_output.squeeze(0))
        gesture_tensor = torch.stack(feats).detach().to(torch.float32)
        self.register_buffer("gesture_features", gesture_tensor)
        del clip_model
        del tokenizer

    def forward(self, spatial_features: torch.Tensor, masks: Optional[torch.Tensor] = None, return_attn: bool = False):
        """
        Args:
            spatial_features: (B, T, D)
            masks: optional (B, T) bool where True=valid
        Returns:
            logits: (B, 1)
        """
        b, t, d = spatial_features.shape
        x = spatial_features

        key_padding_mask = None
        if masks is not None:
            masks = masks.to(torch.bool)
            key_padding_mask = ~masks

        # pad/truncate to n_frames (and pad mask accordingly)
        if t < self.n_frames:
            pad_len = self.n_frames - t
            feat_pad = torch.zeros(b, pad_len, d, device=x.device, dtype=x.dtype)
            x = torch.cat([x, feat_pad], dim=1)
            if key_padding_mask is not None:
                mask_pad = torch.ones(b, pad_len, device=key_padding_mask.device, dtype=torch.bool)
                key_padding_mask = torch.cat([key_padding_mask, mask_pad], dim=1)
            else:
                key_padding_mask = torch.zeros(b, self.n_frames, device=x.device, dtype=torch.bool)
                key_padding_mask[:, t:] = True
        elif t > self.n_frames:
            x = x[:, : self.n_frames, :]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, : self.n_frames]

        x = self.projection(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        gesture_features = self.prompt_act(self.prompt_proj(self.gesture_features))
        gesture_query = gesture_features.unsqueeze(0).expand(b, -1, -1)
        # Dividing the query by the temperature scales the attention logits by
        # 1/temperature, so values < 1 sharpen the prompt-to-frame attention.
        attn_output, attn_weights = self.trans(
            query=gesture_query / self.attn_temperature,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attn_output = self.layer_norm1(attn_output + gesture_query)
        ff_output = self.feed_forward(attn_output)
        ff_output = self.act(ff_output)
        ff_output = self.dropout(ff_output)
        spatial_aware_gestures = self.layer_norm2(ff_output + attn_output)

        gesture_expanded = gesture_features.unsqueeze(0).expand(b, -1, -1)
        refined_ct, _ = self.attention(
            query=spatial_aware_gestures,
            key=gesture_expanded,
            value=gesture_expanded,
        )

        # Pooled video summary (needed unless fusion is plain concat with pooled disabled)
        pooled = None
        if self.fusion != "concat" or not self.disable_pooled:
            pooled = self.video_fc(self._pool_sequence(x, key_padding_mask))

        if self.fusion == "film":
            gamma, beta = self.film_proj(pooled).chunk(2, dim=-1)
            refined_ct = refined_ct * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        ct = refined_ct.reshape(b, -1)
        ct = self.dropout(ct)

        if self.fusion == "gated":
            mean_ct = refined_ct.mean(dim=1)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([mean_ct, pooled], dim=-1)))
            fused = gate * pooled + (1.0 - gate) * mean_ct
            out = self.pred(torch.cat([ct, fused], dim=1))
        elif self.disable_pooled:
            out = self.pred(ct)
        else:
            out = self.pred(torch.cat([ct, pooled], dim=1))
        if return_attn:
            return out, attn_weights
        return out

class GVRModuleContexPred(nn.Module):
    """
    Per-frame context prediction model that consumes TWO pre-extracted feature streams:
      - base_features: (B,T,Db)  (e.g., embed.npy or CLIP-extracted features)
      - ges_features:  (B,T,Dg)  (e.g., ges_embed.npy or tri_embed.npy; later LoRA-CLIP gesture features)

    Output: logits (B, n_frames) (per-frame), matching root `GVRModuleContexPredDualFeatures`.
    """

    def __init__(
        self,
        base_dim: int = 2048,
        ges_dim: int = 2048,
        d_embed: int = 2048,
        num_heads: int = 4,
        n_frames: int = 40,
        layer_norm1: bool = False,
        layer_norm2: bool = True,
        layer_norm3: bool = False,
        position: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_frames = int(n_frames)
        self.use_positional_encoding = position
        self.act = nn.GELU()
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_embed, max_len=self.n_frames)

        self.proj_base = nn.Linear(base_dim, d_embed)
        self.proj_ges = nn.Linear(ges_dim, d_embed)

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

    def forward(self, base_features: torch.Tensor, ges_features: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = base_features.shape
        t_min = min(base_features.size(1), ges_features.size(1))
        base_features = base_features[:, :t_min, :]
        ges_features = ges_features[:, :t_min, :]
        if masks is not None:
            masks = masks[:, :t_min].to(torch.bool)

        # pad/truncate to n_frames
        if t_min < self.n_frames:
            pad_len = self.n_frames - t_min
            base_pad = torch.zeros(b, pad_len, base_features.size(-1), device=base_features.device, dtype=base_features.dtype)
            ges_pad = torch.zeros(b, pad_len, ges_features.size(-1), device=ges_features.device, dtype=ges_features.dtype)
            base_features = torch.cat([base_features, base_pad], dim=1)
            ges_features = torch.cat([ges_features, ges_pad], dim=1)
            if masks is not None:
                mask_pad = torch.zeros(b, pad_len, device=masks.device, dtype=torch.bool)
                masks = torch.cat([masks, mask_pad], dim=1)
        elif t_min > self.n_frames:
            base_features = base_features[:, : self.n_frames, :]
            ges_features = ges_features[:, : self.n_frames, :]
            if masks is not None:
                masks = masks[:, : self.n_frames]

        base_emb = self.act(self.proj_base(base_features))
        base_emb = self.dropout(base_emb)
        ges_emb = self.act(self.proj_ges(ges_features))
        ges_emb = self.dropout(ges_emb)
        if self.use_positional_encoding:
            base_emb = self.positional_encoding(base_emb)
            ges_emb = self.positional_encoding(ges_emb)

        key_padding_mask = None
        if masks is not None:
            key_padding_mask = ~masks.to(torch.bool)

        # Cross-attention: query=gesture branch, key/value=base branch
        attout, _ = self.trans(query=ges_emb, key=base_emb, value=base_emb, key_padding_mask=key_padding_mask)
        if self.layer_norm1_exist:
            attout = self.layer_norm1(attout + ges_emb)

        spatial_aware, _ = self.attention(query=attout, key=ges_emb, value=ges_emb, key_padding_mask=key_padding_mask)
        if self.layer_norm2_exist:
            spatial_aware = self.layer_norm2(spatial_aware + attout)
        if self.layer_norm3_exist:
            spatial_aware = self.layer_norm3(spatial_aware + base_emb)

        spatial_aware = self.dropout(spatial_aware)
        return self.pred(spatial_aware).squeeze(-1)  # (B, T)

class GVRModulePred_kin(nn.Module):
    def __init__(self, gesture_prompts, d_text, d_model=2048, num_heads=1, n_frames=40,position=False):
        super(GVRModulePred, self).__init__()
        # NOTE: legacy class; not used by current SAR_RARP50 feature-based training.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_positional_encoding = position
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_text, max_len=n_frames)
        # CLIP Text Encoder for gesture prompts
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Process the predefined gestures using the CLIP text encoder
        self.gesture_prompts = gesture_prompts
        self.gesture_features = self._generate_gesture_prompt_features(d_text)
       
        self.cnn = models.resnet50(pretrained=True).to(self.device)
        # Remove the final FC layer
        self.cnn.fc = nn.Identity()
        self.cnn_out_dim = d_model
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Linear layers and normalization
        self.projection = nn.Linear(d_model+14, d_text).to(self.device) # add kin dimension
        self.feed_forward = nn.Linear(d_text, d_text).to(self.device)
        self.layer_norm1 = nn.LayerNorm(d_text).to(self.device)
        self.layer_norm2 = nn.LayerNorm(d_text).to(self.device)
        # First Attention layer for spatially-aware gesture features
        self.trans = nn.MultiheadAttention(embed_dim=d_text, num_heads=num_heads, batch_first=True).to(self.device)
        
        # Second Attention layer for refined gesture features
        self.attention = nn.MultiheadAttention(embed_dim=d_text, num_heads=1, batch_first=True).to(self.device)
        self.num_gestures = len(gesture_prompts) 
        self.bottleneck = nn.Linear(self.num_gestures * d_text, 256).to(self.device)
        self.n_frames = n_frames

        self.pred = nn.Linear(self.num_gestures * d_text, 1).to(self.device)

    def _generate_gesture_prompt_features(self, d_text):
        """
        Generate gesture prompt features {gj} using CLIP's text encoder.
        """
        gesture_features = []
        for gesture in self.gesture_prompts:
            prompt = f"{gesture}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  
            gesture_feature = self.clip_model(**inputs).pooler_output.squeeze(0).to(self.device)  
            gesture_features.append(gesture_feature)
        return torch.stack(gesture_features).detach()  # Shape: (J, d_text)

    def forward(self, video_frames,kine):
        batch_size, _, _, _, _ = video_frames.shape
    
        
        # Generate spatial embeddings for the video frames using ResNet50
        # Reshape video frames to process each frame individually
        frames_reshaped = video_frames[:, :self.n_frames].reshape(-1, video_frames.size(2), video_frames.size(3), video_frames.size(4))
        spatial_embeddings = self.cnn(frames_reshaped).view(batch_size, self.n_frames, -1) 
        combined = torch.cat((spatial_embeddings, kine), dim=-1)

        # Project spatial embeddings to d_text dimension
        spatial_embeddings = self.projection(combined)  
        # spatial_embeddings = self.dropout(spatial_embeddings)
        if self.use_positional_encoding:
            spatial_embeddings = self.positional_encoding(spatial_embeddings)
        
        spatial_aware_gestures = []
        
        # Apply first attention with each gesture as query and video features as key/value
        for j in range(self.num_gestures):
            # Gesture feature as query
            gesture_query = self.gesture_features[j].unsqueeze(0).expand(batch_size, -1).unsqueeze(1)  # Shape: (batch_size, 1, d_text)
            
            # Apply attention with gesture as query and spatial embeddings as key/value
            attn_output, _ = self.trans(
                query=gesture_query,  # Shape: (batch_size, 1, d_text)
                key=spatial_embeddings,  # Shape: (batch_size, n_frames, d_text)
                value=spatial_embeddings  # Shape: (batch_size, n_frames, d_text)
            )  # Shape: (batch_size, 1, d_text)
            
            # Remove the singleton dimension
            attn_output = attn_output.squeeze(1)  # Shape: (batch_size, d_text)
            
            # Apply first layer norm and residual connection
            attn_output = self.layer_norm1(attn_output + gesture_query.squeeze(1))
            # Apply feed forward and second layer norm
            ff_output = self.feed_forward(attn_output)
            final_output = self.layer_norm2(ff_output + attn_output)
            spatial_aware_gestures.append(final_output) 
        
        # Stack spatial-aware gesture features to prepare for secondary attention
        spatial_aware_gestures = torch.stack(spatial_aware_gestures, dim=1)  
        # Apply secondary attention with spatial-aware gestures as query and gesture features as key/value
        refined_ct, _ = self.attention(
            query=spatial_aware_gestures,  # Shape: (batch_size, J, d_text)
            key=self.gesture_features.unsqueeze(0).expand(batch_size, -1, -1),  # Original gesture features as key
            value=self.gesture_features.unsqueeze(0).expand(batch_size, -1, -1)  # Original gesture features as value
        )  # Shape: (batch_size, J, d_text)
        
        # Flatten the refined gestures into a single feature vector
        ct = refined_ct.reshape(batch_size, -1)  # Flattened feature vector 
        # Final prediction
        out = self.pred(ct) 
        
        return out