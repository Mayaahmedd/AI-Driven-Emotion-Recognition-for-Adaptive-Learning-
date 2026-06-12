"""
Model for exp39.

This is a faithful copy of the exp38 architecture:

    ResNet50  ->  SE channel attention  ->  2-layer BiLSTM  ->
    Temporal attention pooling  ->  MLP head  ->  4 multilabel logits

Differences vs exp38
--------------------
1. The class is renamed (``...Exp39``) so both experiments can be imported
   side by side without symbol collisions.
2. The ResNet50 ImageNet weights variant is named explicitly
   (``ResNet50_Weights.IMAGENET1K_V2``). exp38 used ``DEFAULT`` which silently
   resolves to V2 today but could change with a future torchvision release;
   pinning the variant removes this implicit dependency from the methodology.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block operating on per-frame feature vectors.

    Input:  ``[B, T, D]`` features
    Output: ``[B, T, D]`` features with learned channel re-weighting.
    """

    def __init__(self, feature_dim: int, reduction: int = 16):
        super().__init__()
        hidden_dim = max(feature_dim // reduction, 64)
        self.se = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.se(x)
        return x * weights


class TemporalAttention(nn.Module):
    """Single-head additive temporal attention over T time steps."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        scores = self.attention(x)            # B, T, 1
        weights = torch.softmax(scores, dim=1) # B, T, 1
        context = torch.sum(weights * x, dim=1)  # B, D
        return context, weights


class ResNet50SEBiLSTM2AttentionMultilabelExp39(nn.Module):
    """ResNet50 + SE + 2-layer BiLSTM + Temporal Attention + Multilabel head.

    Parameters
    ----------
    num_labels
        Number of independent binary heads (DAiSEE → 4).
    pretrained
        If True, loads ``ResNet50_Weights.IMAGENET1K_V2`` weights explicitly.
    hidden_size
        Hidden size per direction of the BiLSTM (default 256).
    num_layers
        Number of stacked BiLSTM layers (default 2).
    dropout
        Dropout used inside the LSTM stack and in the classification head.
    freeze_backbone
        If True, freezes ResNet50 parameters (transfer-learning mode).
    """

    def __init__(
        self,
        num_labels: int = 4,
        pretrained: bool = True,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
        backbone = models.resnet50(weights=weights)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.se_block = SEBlock1D(self.feature_dim, reduction=16)

        self.bilstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.temporal_dim = hidden_size * 2
        self.temporal_attention = TemporalAttention(self.temporal_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.temporal_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: B, T, 3, H, W
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.feature_extractor(x).flatten(1)     # B*T, 2048
        feat = feat.view(b, t, self.feature_dim)         # B, T, 2048
        feat = self.se_block(feat)
        lstm_out, _ = self.bilstm(feat)                  # B, T, 2*hidden
        video_feat, attn_weights = self.temporal_attention(lstm_out)
        logits = self.classifier(video_feat)             # B, num_labels
        if return_attention:
            return logits, attn_weights
        return logits
