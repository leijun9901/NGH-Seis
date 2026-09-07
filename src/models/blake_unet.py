"""Light anisotropic U-Net baseline for PSDM-to-Sh/Sg regression."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class _Block(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class BlakeLightUNet(nn.Module):
    """A compact same-grid image-to-image baseline; mask is not an input."""

    def __init__(self, base_channels: int = 12) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = _Block(1, c)
        self.enc2 = _Block(c, 2 * c)
        self.enc3 = _Block(2 * c, 4 * c)
        self.enc4 = _Block(4 * c, 8 * c)
        self.bridge = _Block(8 * c, 12 * c)
        self.dec4 = _Block(12 * c + 8 * c, 8 * c)
        self.dec3 = _Block(8 * c + 4 * c, 4 * c)
        self.dec2 = _Block(4 * c + 2 * c, 2 * c)
        self.dec1 = _Block(2 * c + c, c)
        self.head = nn.Conv2d(c, 2, 1)
        # Saturation occupies a small part of the image.  A low-probability
        # prior avoids beginning optimization with 50% saturation everywhere.
        nn.init.constant_(self.head.bias, -4.0)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(f"Expected [B,1,H,W] with H/W divisible by 16, got {tuple(x.shape)}")
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        bridge = self.bridge(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self._up(bridge, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return torch.sigmoid(self.head(d1))


def blake_saturation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    background_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sparse-aware normalized regression loss with edge preservation."""
    thresholds = target.new_tensor([0.01 / 0.25, 0.003 / 0.10])[None, :, None, None]
    support = torch.clamp(target / thresholds, 0.0, 1.0)
    weights = background_weight + (1.0 - background_weight) * support
    valid = mask.expand_as(target)
    point = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.03)
    regression = (point * weights * valid).sum() / (weights * valid).sum().clamp_min(1.0)

    grad_pred_z = prediction[:, :, 1:] - prediction[:, :, :-1]
    grad_true_z = target[:, :, 1:] - target[:, :, :-1]
    grad_mask_z = valid[:, :, 1:] * valid[:, :, :-1]
    gradient = ((grad_pred_z - grad_true_z).abs() * grad_mask_z).sum() / grad_mask_z.sum().clamp_min(1.0)
    total = regression + 0.05 * gradient
    return total, {
        "loss": total.detach(),
        "regression": regression.detach(),
        "gradient": gradient.detach(),
    }
