
import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock1D(nn.Module):
    def __init__(self, feature_dim, reduction=16):
        super().__init__()

        hidden_dim = max(feature_dim // reduction, 64)

        self.se = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B,T,D
        weights = self.se(x)
        return x * weights


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: B,T,D
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


class ResNet50SEBiLSTM2AttentionMultilabel(nn.Module):
    def __init__(
        self,
        num_labels=4,
        pretrained=True,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        freeze_backbone=False,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
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

    def forward(self, x, return_attention=False):
        # x: B,T,C,H,W
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)

        feat = self.feature_extractor(x)
        feat = feat.flatten(1)
        feat = feat.view(b, t, self.feature_dim)

        feat = self.se_block(feat)

        lstm_out, _ = self.bilstm(feat)

        video_feat, attn_weights = self.temporal_attention(lstm_out)

        logits = self.classifier(video_feat)

        if return_attention:
            return logits, attn_weights

        return logits
