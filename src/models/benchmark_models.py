"""Representative segmentation/regression backbones for both benchmark tasks."""

from __future__ import annotations

import math

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


class _DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, dim, kernel_size=kernel, stride=stride, padding=kernel // 2
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class _EfficientAttention(nn.Module):
    def __init__(self, dim: int, heads: int, reduction: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Attention dimension must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction, stride=reduction)
            self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q = self.query(x).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        reduced = x
        if self.reduction > 1:
            reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.sr(reduced).reshape(batch, channels, -1).transpose(1, 2)
            reduced = self.norm(reduced)
        kv = self.key_value(reduced).reshape(
            batch, reduced.shape[1], 2, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]
        attention = (q @ key.transpose(-2, -1)) * self.scale
        output = (attention.softmax(dim=-1) @ value).transpose(1, 2).reshape(
            batch, tokens, channels
        )
        return self.projection(output)


class _MixFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, _ = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = F.gelu(self.depthwise(x))
        x = x.flatten(2).transpose(1, 2)
        return self.fc2(x)


class _TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, reduction: int, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = _EfficientAttention(dim, heads, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _MixFFN(dim)
        self.drop_path = _DropPath(drop_path)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        return x + self.drop_path(self.ffn(self.norm2(x), height, width))


class SegFormerB0(nn.Module):
    """SegFormer-B0 encoder and all-MLP decoder for dense prediction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_activation: str = "identity",
        decoder_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.output_activation = output_activation
        dims = (32, 64, 160, 256)
        heads = (1, 2, 5, 8)
        reductions = (8, 4, 2, 1)
        depths = (2, 2, 2, 2)
        kernels = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        drop_rates = torch.linspace(0.0, 0.1, sum(depths)).tolist()
        offset = 0
        previous = in_channels
        self.patch_embeddings = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        for dim, head, reduction, depth, kernel, stride in zip(
            dims, heads, reductions, depths, kernels, strides
        ):
            self.patch_embeddings.append(
                _OverlapPatchEmbed(previous, dim, kernel=kernel, stride=stride)
            )
            self.stages.append(
                nn.ModuleList(
                    [
                        _TransformerBlock(
                            dim, head, reduction, drop_path=drop_rates[offset + index]
                        )
                        for index in range(depth)
                    ]
                )
            )
            self.stage_norms.append(nn.LayerNorm(dim))
            previous = dim
            offset += depth
        self.projections = nn.ModuleList(
            [nn.Conv2d(dim, decoder_dim, 1) for dim in dims]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.head = nn.Conv2d(decoder_dim, out_channels, 1)
        self.apply(self._initialize)
        _initialize_head(self.head, output_activation)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected Bx{self.in_channels}xHxW, got {tuple(x.shape)}")
        output_size = x.shape[-2:]
        features = []
        for embedding, blocks, norm in zip(
            self.patch_embeddings, self.stages, self.stage_norms
        ):
            tokens, height, width = embedding(x)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = norm(tokens)
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
            features.append(x)
        target_size = features[0].shape[-2:]
        decoded = [
            F.interpolate(projection(feature), size=target_size, mode="bilinear", align_corners=False)
            for projection, feature in zip(self.projections, features)
        ]
        output = self.head(self.fuse(torch.cat(decoded, dim=1)))
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        return _activate(output, self.output_activation)


MODEL_NAMES = ("unet", "resunet", "deeplabv3plus", "segformer_b0")


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
    if name == "segformer_b0":
        return SegFormerB0(in_channels, out_channels, output_activation)
    raise ValueError(f"Unknown benchmark model {name!r}; choose from {MODEL_NAMES}")
