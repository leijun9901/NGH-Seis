"""Representative segmentation/regression backbones for both benchmark tasks."""

from __future__ import annotations


import torch
from torch import nn
from torch.nn import functional as F

from .standard_unet import StandardUNet


def _activate(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"Unknown output activation {name!r}")


def _initialize_head(head: nn.Conv2d, activation: str) -> None:
    nn.init.kaiming_normal_(head.weight, mode="fan_in", nonlinearity="linear")
    nn.init.constant_(head.bias, -4.0 if activation == "sigmoid" else 0.0)


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class ResUNet(nn.Module):
    """Four-level residual U-Net with the same resolution hierarchy as U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_activation = output_activation
        c = int(base_channels)
        self.enc1 = _ResidualBlock(in_channels, c)
        self.enc2 = _ResidualBlock(c, 2 * c)
        self.enc3 = _ResidualBlock(2 * c, 4 * c)
        self.enc4 = _ResidualBlock(4 * c, 8 * c)
        self.bridge = _ResidualBlock(8 * c, 16 * c)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(16 * c, 8 * c, 2, stride=2)
        self.dec4 = _ResidualBlock(16 * c, 8 * c)
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = _ResidualBlock(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = _ResidualBlock(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = _ResidualBlock(2 * c, c)
        self.head = nn.Conv2d(c, out_channels, 1)
        _initialize_head(self.head, output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected Bx{self.in_channels}xHxW, got {tuple(x.shape)}")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("ResUNet requires H/W divisible by 16")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bridge = self.bridge(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bridge), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return _activate(self.head(d1), self.output_activation)


class _ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                1 if dilation == 1 else 3,
                padding=0 if dilation == 1 else dilation,
                dilation=1 if dilation == 1 else dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [_ASPPConv(in_channels, out_channels, rate) for rate in (1, 6, 12, 18)]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = F.interpolate(
            self.image_pool(x), size=size, mode="bilinear", align_corners=False
        )
        return self.project(torch.cat([*(branch(x) for branch in self.branches), pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a standard output-stride-16 ResNet-50 encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_activation: str = "identity",
    ) -> None:
        super().__init__()
        from torchvision.models import resnet50

        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.in_channels = int(in_channels)
        self.output_activation = output_activation
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = _ASPP(2048, 256)
        self.low_projection = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, out_channels, 1)
        _initialize_head(self.head, output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected Bx{self.in_channels}xHxW, got {tuple(x.shape)}")
        output_size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low_projection(low)], dim=1))
        x = self.head(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return _activate(x, self.output_activation)


MODEL_NAMES = ("unet", "resunet", "deeplabv3plus")


def build_benchmark_model(
    name: str,
    *,
    in_channels: int,
    out_channels: int = 2,
    base_channels: int = 32,
    output_activation: str,
) -> nn.Module:
    if name == "unet":
        return StandardUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            output_activation=output_activation,
        )
    if name == "resunet":
        return ResUNet(in_channels, out_channels, base_channels, output_activation)
    if name == "deeplabv3plus":
        return DeepLabV3Plus(in_channels, out_channels, output_activation)
    raise ValueError(f"Unknown benchmark model {name!r}; choose from {MODEL_NAMES}")
