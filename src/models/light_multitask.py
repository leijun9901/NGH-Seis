"""A compact gather-to-Sh/Sg baseline.

Vp belongs to the synthetic forward chain, not to this network's outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class LightHFGNet(nn.Module):
    """Compact baseline; input [B,8,320,128], Sh/Sg output [B,2,200,200]."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            ConvBlock(8, c, stride=2),       # 160 x 64
            ConvBlock(c, 2 * c, stride=2),   # 80 x 32
            ConvBlock(2 * c, 3 * c, stride=2),  # 40 x 16
            ConvBlock(3 * c, 4 * c, stride=2),  # 20 x 8
        )
        self.bridge = ConvBlock(4 * c, 4 * c)
        self.decode_50 = ConvBlock(4 * c, 3 * c)
        self.decode_100 = ConvBlock(3 * c, 2 * c)
        self.decode_200 = ConvBlock(2 * c, c)
        self.saturation_head = nn.Conv2d(c, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (8, 320, 128):
            raise ValueError(f"Expected [B,8,320,128], got {tuple(x.shape)}")
        x = self.encoder(x)
        # Learned convolutions after this coordinate conversion map acquisition
        # coordinates (time/receiver) to model coordinates (depth/lateral).
        x = F.interpolate(x, size=(25, 25), mode="bilinear", align_corners=False)
        x = self.bridge(x)
        x = self.decode_50(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.decode_100(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.decode_200(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return torch.sigmoid(self.saturation_head(x))


def saturation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    saturation_background_weight: float = 0.08,
    phase_penalty_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sparse-aware Sh/Sg regression loss with a pore-volume constraint."""
    active = (target.sum(dim=1, keepdim=True) > 1e-4).to(target.dtype)
    sat_weights = saturation_background_weight + (1.0 - saturation_background_weight) * active
    sat_map = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.02)
    sat_loss = (sat_map * sat_weights).sum() / (sat_weights.sum() * target.shape[1])

    # The two pore-fluid saturations cannot occupy more than the pore volume.
    phase_penalty = F.relu(prediction.sum(dim=1) - 1.0).mean()
    total = sat_loss + phase_penalty_weight * phase_penalty
    return total, {
        "loss": total.detach(),
        "sat_loss": sat_loss.detach(),
        "phase_penalty": phase_penalty.detach(),
    }
