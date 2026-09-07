import pytest


torch = pytest.importorskip("torch")

from src.models.standard_unet import StandardUNet


def test_standard_unet_preserves_saturation_interface():
    model = StandardUNet(base_channels=4)
    output = model(torch.randn(1, 1, 32, 32))
    assert output.shape == (1, 2, 32, 32)
    assert bool(((output >= 0.0) & (output <= 1.0)).all())


def test_standard_unet_supports_two_channel_continuous_regression():
    model = StandardUNet(
        in_channels=2, out_channels=2, base_channels=4, output_activation="identity"
    )
    output = model(torch.randn(1, 2, 32, 32))
    assert output.shape == (1, 2, 32, 32)
    assert bool(torch.isfinite(output).all())
