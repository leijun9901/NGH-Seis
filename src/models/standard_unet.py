"""A conventional 2-D U-Net baseline with no task-specific input branches."""

from __future__ import annotations

import torch
from torch import nn


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class StandardUNet(nn.Module):
    """Four-level U-Net: Conv-BN-ReLU blocks and transposed-conv decoding.

    ``output_activation="sigmoid"`` preserves the original saturation
    baseline.  ``output_activation="identity"`` provides the same backbone
    for continuous, standardized regression targets without adding any
    task-specific branch.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 32,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if output_activation not in {"sigmoid", "identity"}:
            raise ValueError(f"Unknown output activation {output_activation!r}")
        self.in_channels = int(in_channels)
        self.output_activation = output_activation
        c = int(base_channels)
        self.enc1 = _DoubleConv(in_channels, c)
        self.enc2 = _DoubleConv(c, 2 * c)
        self.enc3 = _DoubleConv(2 * c, 4 * c)
        self.enc4 = _DoubleConv(4 * c, 8 * c)
        self.bridge = _DoubleConv(8 * c, 16 * c)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, stride=2)
        self.dec4 = _DoubleConv(16 * c, 8 * c)
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = _DoubleConv(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = _DoubleConv(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = _DoubleConv(2 * c, c)
        self.head = nn.Conv2d(c, out_channels, 1)
        nn.init.constant_(self.head.bias, -4.0 if output_activation == "sigmoid" else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.ndim != 4
            or x.shape[1] != self.in_channels
            or x.shape[-2] % 16
            or x.shape[-1] % 16
        ):
            raise ValueError(
                f"Expected [B,{self.in_channels},H,W], H/W divisible by 16; "
                f"got {tuple(x.shape)}"
            )
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bridge(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        output = self.head(d1)
        return torch.sigmoid(output) if self.output_activation == "sigmoid" else output
