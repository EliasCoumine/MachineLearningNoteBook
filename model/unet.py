"""
UNet – Aerial House Segmentation
CEG4195/SEG4180 · uOttawa Winter 2026

Architecture
────────────
Standard U-Net (Ronneberger et al. 2015) with:
  • Double-convolution blocks at each resolution level.
  • Skip connections between encoder and decoder.
  • Optional dropout for regularisation.

We also expose a factory function that wraps segmentation-models-pytorch's
UNet with a pretrained ResNet-34 encoder for transfer learning.

Usage
─────
    from model.unet import UNet, build_transfer_unet

    # From scratch
    model = UNet(in_channels=3, out_channels=1, base_filters=64)

    # Transfer learning (recommended)
    model = build_transfer_unet(encoder="resnet34", pretrained=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv→BN→ReLU blocks (the basic UNet unit)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Bilinear upsample → concat skip → DoubleConv (decoder step).

    Bilinear upsampling is chosen over transposed convolutions for stability
    and to avoid checkerboard artefacts.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip dimensions in case of odd input sizes
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x  = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full UNet (trained from scratch)
# ─────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net for binary semantic segmentation.

    Parameters
    ----------
    in_channels  : Number of input channels (3 for RGB).
    out_channels : Number of output channels (1 for binary mask).
    base_filters : Number of filters in the first encoder block; doubled at
                   each level.
    dropout      : Dropout probability applied inside DoubleConv blocks.

    Forward pass
    ------------
    Input  : (B, in_channels, H, W)
    Output : (B, out_channels, H, W)  – raw logits (apply sigmoid externally)
    """

    def __init__(
        self,
        in_channels:  int   = 3,
        out_channels: int   = 1,
        base_filters: int   = 64,
        dropout:      float = 0.0,
    ):
        super().__init__()
        f = base_filters

        # Encoder
        self.inc   = DoubleConv(in_channels, f,      dropout)
        self.down1 = Down(f,      f * 2,  dropout)
        self.down2 = Down(f * 2,  f * 4,  dropout)
        self.down3 = Down(f * 4,  f * 8,  dropout)
        self.down4 = Down(f * 8,  f * 16, dropout)   # bottleneck

        # Decoder
        self.up1   = Up(f * 16 + f * 8,  f * 8,  dropout)
        self.up2   = Up(f * 8  + f * 4,  f * 4,  dropout)
        self.up3   = Up(f * 4  + f * 2,  f * 2,  dropout)
        self.up4   = Up(f * 2  + f,      f,       dropout)

        # Output projection
        self.outc  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transfer-learning UNet via segmentation-models-pytorch
# ─────────────────────────────────────────────────────────────────────────────

def build_transfer_unet(
    encoder:    str  = "resnet34",
    pretrained: bool = True,
    in_channels: int = 3,
    classes:     int = 1,
):
    """
    Build a UNet with a pretrained ImageNet encoder (transfer learning).

    This is the recommended model for training, as the pretrained encoder
    provides a substantial head-start over random initialisation and reduces
    the amount of training data required.

    Parameters
    ----------
    encoder     : Any encoder supported by smp (e.g. 'resnet34', 'efficientnet-b3').
    pretrained  : Use ImageNet pretrained weights for the encoder.
    in_channels : Input image channels.
    classes     : Number of output channels (1 for binary segmentation).

    Returns
    -------
    model : smp.Unet instance ready for training.
    """
    import segmentation_models_pytorch as smp

    weights = "imagenet" if pretrained else None
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,    # raw logits; apply sigmoid in loss / post-processing
    )
