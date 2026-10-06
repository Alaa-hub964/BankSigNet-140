"""
MSFV-Net: Multi-Script Forgery Verification Network
=====================================================
Paper: MSFV-Net: Explainable Multi-Script Offline Signature Verification
       on Real Bank Cheque Data
Dataset: BankSigNet-140
Authors: Alaa Alowaidi, Pushpendra Kumar Pateriya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MSFVNet(nn.Module):
    """
    MSFV-Net: ResNet50-based Siamese verification network.

    Architecture:
        encoder.features : ResNet50 body (grayscale input, 1-channel conv1)
        encoder.fc       : 2048 -> 1024 -> 512 (L2-normalised embedding)
        head             : [e1, e2, |e1-e2|, e1*e2] -> 512 -> 128 -> 1

    Forward:
        forward_one(x)          : image -> 512-d embedding
        forward_pair(e1, e2)    : pair of embeddings -> logit
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()

        #  Encoder
        backbone = models.resnet50(weights=None)
        # Grayscale input (1 channel)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = nn.ModuleDict(
            {
                "features": nn.Sequential(*list(backbone.children())[:-1]),
                "fc": nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(1024, embed_dim),
                ),
            }
        )

        # Verification head
        # Input: [e1, e2, |e1-e2|, e1*e2] = 4 * embed_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single signature image to a unit-normalised embedding.

        Args:
            x: (B, 1, 224, 224) grayscale signature image

        Returns:
            embedding: (B, 512) L2-normalised
        """
        x = self.encoder["features"](x)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = self.encoder["fc"](x)  # (B, 512)
        return F.normalize(x, dim=1)

    def forward_pair(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        Compute genuine/forged logit for a pair of embeddings.

        Args:
            e1: (B, 512) query embedding
            e2: (B, 512) reference embedding

        Returns:
            logit: (B,) — apply sigmoid for probability
        """
        combined = torch.cat([e1, e2, (e1 - e2).abs(), e1 * e2], dim=1)  # (B, 2048)
        return self.head(combined).squeeze(1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass for a pair of images."""
        e1 = self.forward_one(x1)
        e2 = self.forward_one(x2)
        return self.forward_pair(e1, e2)


def get_transform():
    """Standard image transform for BankSigNet-140."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def load_model(checkpoint_path: str, device: torch.device = None) -> MSFVNet:
    """
    Load a trained MSFV-Net checkpoint.

    Args:
        checkpoint_path: path to .pth file
        device: torch device (default: cuda if available)

    Returns:
        model: MSFVNet in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MSFVNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
