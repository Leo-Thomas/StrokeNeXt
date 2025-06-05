"""
StrokeNeXt: A Siamese-encoder Approach for Brain Stroke Classification in Computed Tomography Imagery

This module defines the StrokeNeXt architecture, composed of:
- Two parallel encoders (e.g., ConvNeXt, ResNet) from torchvision
- A lightweight 1D convolutional fusion module
- A compact fully-connected classifier

Usage:
    model = StrokeNeXt(
        n_classes=num_classes,
        encoder_name='convnext_tiny',
        feature_dim=768,
        fusion_hidden=None,
        fusion_dropout=0.5,
        classifier_dropout=0.5
    )

Author: Leo Thomas Ramos, Computer Vision Center, Universitat Autònoma de Barcelona, Spain.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Conv1dFusion(nn.Module):

    def __init__(self, feature_dim, hidden_dim=None, dropout_p=0.1):
        super(Conv1dFusion, self).__init__()
        hidden_dim = feature_dim if hidden_dim is None else hidden_dim

        self.conv_reduce = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_reduce = nn.BatchNorm1d(feature_dim)

        self.conv_point = nn.Conv1d(
            in_channels=feature_dim, out_channels=hidden_dim, kernel_size=1, bias=False
        )
        self.bn_point = nn.BatchNorm1d(hidden_dim)

        if hidden_dim != feature_dim:
            self.conv_expand = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=feature_dim,
                kernel_size=1,
                bias=False,
            )
            self.bn_expand = nn.BatchNorm1d(feature_dim)
        else:
            self.conv_expand = None

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, f1, f2):

        x = torch.stack([f1, f2], dim=2)

        x = self.conv_reduce(x)
        x = x.squeeze(2)
        x = self.bn_reduce(x)
        x = self.gelu(x)

        x = x.unsqueeze(2)
        x = self.conv_point(x)
        x = x.squeeze(2)
        x = self.bn_point(x)
        x = self.gelu(x)
        x = self.dropout(x)

        if self.conv_expand is not None:
            x = x.unsqueeze(2)
            x = self.conv_expand(x)
            x = x.squeeze(2)
            x = self.bn_expand(x)
            x = self.gelu(x)

        return x


class StrokeNeXt(nn.Module):

    def __init__(
        self,
        n_classes: int,
        encoder_name: str = "convnext_tinyy",  # convnext_small, convnext_base, convnext_large
        feature_dim: int = 768,  # 768, 1024, 1536
        fusion_hidden: int = None,
        fusion_dropout: float = 0.5,
        classifier_dropout: float = 0.5,
    ):
        super(StrokeNeXt, self).__init__()

        if not hasattr(models, encoder_name):
            raise ValueError(
                f"Encoder '{encoder_name}' not found in torchvision.models"
            )

        encoder_fn = getattr(models, encoder_name)

        base1 = encoder_fn(pretrained=True)
        self.encoder1 = nn.Sequential(*list(base1.children())[:-1])

        base2 = encoder_fn(pretrained=True)
        self.encoder2 = nn.Sequential(*list(base2.children())[:-1])

        self.fusion = Conv1dFusion(
            feature_dim=feature_dim, hidden_dim=fusion_hidden, dropout_p=fusion_dropout
        )

        hidden_dim = 256
        # hidden_dim = feature_dim // 2

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(hidden_dim, n_classes, bias=True),
        )

    def forward(self, input1, input2):
        x1 = self.encoder1(input1)
        f1 = x1.view(x1.size(0), -1)

        x2 = self.encoder2(input2)
        f2 = x2.view(x2.size(0), -1)

        fused = self.fusion(f1, f2)

        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    model = StrokeNeXt(n_classes=2)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input, dummy_input)

    print("Output shape:", output.shape)
