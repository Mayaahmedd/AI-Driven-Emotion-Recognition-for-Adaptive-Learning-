import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50BiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ResNet-50 backbone
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            backbone = resnet50(weights=None)

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Temporal model
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        x shape: [B, T, C, H, W]
        returns: [B, 4]
        """
        b, t, c, h, w = x.shape

        # Merge batch and time so ResNet processes frames one by one
        x = x.view(b * t, c, h, w)

        # Extract frame features
        feats = self.feature_extractor(x)          # [B*T, 2048, 1, 1]
        feats = feats.view(b * t, self.feature_dim)  # [B*T, 2048]

        # Restore sequence shape
        feats = feats.view(b, t, self.feature_dim)   # [B, T, 2048]

        # Temporal modeling
        lstm_out, _ = self.lstm(feats)               # [B, T, 2*hidden]

        # Use the last time step
        seq_repr = lstm_out[:, -1, :]                # [B, 2*hidden]
        seq_repr = self.dropout(seq_repr)

        logits = self.classifier(seq_repr)           # [B, 4]
        return logits